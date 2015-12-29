#define NOMINMAX
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <thread>
#include <stdint.h>
#include <ctime>
#include <concurrent_queue.h>

using namespace std;
using namespace concurrency;

#include <Windows.h>

#define KEY_LEFT 2424832
#define KEY_RIGHT 2555904

bool running = true;

std::vector<double> a_avg;
std::vector<double> a_dev;
unsigned const a_depth = 500;

double masksum_threshold = 0.5e-3;
float alpha_avg = 0.1f;
float alpha_dev = 0.02f;
float sigma_dev = 10.0f;
float sigma_min = 1.0f;
int reduction = 2;
bool blendframes = false;
// TODO: blendframes: reduction using cv::max() on frame group

bool headless = false;

cv::Point2i blur(3, 3);
std::string suffix("");

// TODO: command line switches

std::string videofile;
double override_fps = 0.0;
double vid_fps;
int width, height;
int nframes;

int preview_scale = 3;
cv::Size preview_size(1920 / preview_scale, 1080 / preview_scale);

int audiorate = 8000;
int audiofreq = 1000;

void parse_args(int argc, char **argv)
{
	// TODO: -fps 25 instead of -reduction

	for (int k = 1; k < argc; k += 1)
	{
		std::string arg = argv[k];

		if (arg[0] == '-')
		{
			if (arg == "-alpha_avg")
			{
				alpha_avg = atof(argv[++k]);
				std::cout << "alpha_avg = " << alpha_avg << std::endl;
			}
			else if (arg == "-alpha_dev")
			{
				alpha_dev = atof(argv[++k]);
				std::cout << "alpha_dev = " << alpha_dev << std::endl;
			}
			else if (arg == "-sigma_dev")
			{
				sigma_dev = atof(argv[++k]);
				std::cout << "sigma_dev = " << sigma_dev << std::endl;
			}
			else if (arg == "-threshold")
			{
				masksum_threshold = atof(argv[++k]);
				std::cout << "threshold = " << masksum_threshold << std::endl;
			}
			else if (arg == "-fps")
			{
				override_fps = atof(argv[++k]);
				std::cout << "override fps = " << override_fps << std::endl;
			}
			else if (arg == "-reduction")
			{
				reduction = atoi(argv[++k]);
				std::cout << "reduction = " << reduction << std::endl;
			}
			else if (arg == "-blend")
			{
				blendframes = true;
				std::cout << "blend = " << (blendframes ? "true" : "false") << std::endl;
			}
			else if (arg == "-blur")
			{
				int arg = atoi(argv[++k]);
				blur = cv::Point2i(arg, arg);
				std::cout << "blur = " << arg << "x" << arg << std::endl;
			}
			else if (arg == "-suffix")
			{
				suffix = argv[++k];
				std::cout << "suffix = " << suffix << std::endl;
			}
			else if (arg == "-headless")
			{
				headless = true;
				std::cout << "headless" << std::endl;
			}
			else
			{
				std::cerr << "unknown switch: " << arg << std::endl;
			}
		}
		else
		{
			if (!videofile.empty())
				std::cerr << "WARNING: multiple input files not supported!" << std::endl;

			videofile = arg;
			std::cout << "input file = " << videofile << std::endl;
		}
	}
}

struct MatOrNothing
{
	bool anything;
	cv::Mat frame;
	cv::Mat gray;
	int frameno;
};

struct WriteElement
{
	bool done;
	unsigned frameno;
	cv::Mat mat;
	double masksum;
};


concurrent_queue<MatOrNothing> framequeue;
concurrent_queue<bool> donequeue;
concurrent_queue<WriteElement> writequeue;


double hrtimer()
{
#if _WIN32
	unsigned long long counter, freq;
	QueryPerformanceCounter((LARGE_INTEGER*)&counter);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)counter / (double)freq;
#else
	struct timeval now;
	int const rv = gettimeofday(&now, NULL);

	assert(rv == 0);

	return now.tv_sec + now.tv_usec * 1e-6;
#endif
}


std::string to_hms(double secs)
{
	int hours = (int)(secs / 3600);
	secs = fmod(secs, 3600);
	int minutes = (int)(secs / 60);
	secs = fmod(secs, 60);

	char buf[100];
	sprintf_s(buf, "%d:%02d:%06.3f", hours, minutes, secs);
	return std::string(buf);
}

std::string describe(cv::Mat const &mat)
{
	std::ostringstream out;
	out << mat.rows << "x" << mat.cols;
	out << ", " << mat.channels() << " channels";
	out << ", type " << CV_MAT_DEPTH(mat.type()) << ", " << mat.elemSize1() << " byte elemsize1, " << mat.elemSize() << " byte elemsize";
	return out.str();
}

cv::Mat getChannel(cv::Mat &mat, int channel)
{
	assert(channel >= 0 && channel < mat.channels());
	assert(mat.dims == 2);
	int const sizes[2] = { mat.rows, mat.cols };

	int const type = CV_MAKETYPE(CV_MAT_DEPTH(mat.type()), 1);

	void *data = ((unsigned char*)mat.data) + channel * mat.elemSize1();

	size_t steps[2] = { mat.step[0], mat.step[1] };

	cv::Mat foo(2, sizes, type, data, steps);
	return foo;
}

std::string splitext(std::string path)
{
	size_t index = path.find_last_of(".");

	if (index == std::string::npos)
		return path;

	return path.substr(0, index);
}

template <typename T>
T linmap(T x, T amin, T amax, T bmin, T bmax)
{
	if (x < amin) return bmin;
	if (x > amax) return bmax;
	return (x - amin) / (amax - amin) * (bmax - bmin) + bmin;
}

void readfn(cv::VideoCapture *vid)
{
	int frameno = (int)vid->get(CV_CAP_PROP_POS_FRAMES);
	int const width = (int)vid->get(CV_CAP_PROP_FRAME_WIDTH);
	int const height = (int)vid->get(CV_CAP_PROP_FRAME_HEIGHT);

	cv::Mat blendedframe, curframe;
	cv::Mat floatframe;
	cv::Mat grayframe;

	blendedframe.create(cv::Size(width, height), CV_16UC3);

	while (running)
	{
		if (!vid->grab())
			break;

		bool do_process = true;

		if (blendframes)
		{
			vid->retrieve(curframe);
			//curframe.convertTo(curframe, CV_16UC3);

			/*
			std::ostringstream label;
			label << "Frame " << frameno;
			cv::putText(curframe, label.str(), cv::Point(10, 20 * (frameno % 50)), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 2);
			//*/

			//std::cerr << "%reduction = " << (frameno % reduction) << std::endl;

			if (frameno % reduction == 0)
				blendedframe.setTo(0);

			//blendedframe = cv::max(blendedframe, curframe);
			cv::add(blendedframe, curframe, blendedframe, cv::noArray(), CV_16UC3);

			if (frameno % reduction == reduction - 1)
				blendedframe.convertTo(floatframe, CV_32FC3, 1.0 / (reduction * 255));
			else
				do_process = false;



			/*
			cv::Mat foo;
			cv::resize(blendedframe, foo, cv::Size(640, 360));
			cv::imshow("debug", foo);
			if (cv::waitKey(100) != -1) exit(0);
			//*/

			//std::cerr << "do_process = " << (do_process) << std::endl;

		}
		else
		{
			if (frameno % reduction == reduction - 1)
			{
				vid->retrieve(curframe);
				curframe.convertTo(floatframe, CV_32FC3, 1.0 / 255);
			}
			else
			{
				do_process = false;
			}

		}

		frameno += 1;

		if (!do_process)
			continue;

		// TODO: proper bounded queue...
		while (running)
		{
			bool foo;
			bool res = donequeue.try_pop(foo);
			if (res)
				break;
		}

		// frameu8 is ready
		cv::cvtColor(floatframe, grayframe, CV_BGR2GRAY, 1);
		cv::blur(grayframe, grayframe, blur);

		MatOrNothing const tmp = { true, floatframe.clone(), grayframe.clone(), frameno };
		framequeue.push(tmp);
		//std::cout << "pushing frame " << frameno << std::endl;
	}

	MatOrNothing const tmp = { false };
	framequeue.push(tmp);
}

int iround(double x)
{
	return static_cast<int>(x + 0.5);
}

void writefn(std::string outaudio, std::string outdiff)
{
	std::ofstream oaudio(outaudio, std::ios::binary);
	//int fourcc = CV_FOURCC('4', '6', '2', 'H');
	int fourcc = -1;
	std::cout << "creating VideoWriter\n";
	std::cout << "fps " << vid_fps << ", reduction " << reduction << "\n";
	cv::VideoWriter odiff(outdiff.c_str(), fourcc, vid_fps / reduction, cv::Size(width, height), true);
	//cv::VideoWriter omask(outmask.c_str(), fourcc, vid_fps / reduction, cv::Size(width, height), true);
	std::cout << "done creating VideoWriter\n";

	WriteElement element;

	while (true)
	{
		bool rv = writequeue.try_pop(element);
		if (!rv) continue;

		if (element.done) break;

		// video

		odiff.write(element.mat);

		// audio

		int u = (int)iround(audiorate * (element.frameno + 0) / (vid_fps / reduction));
		int v = (int)iround(audiorate * (element.frameno + 1) / (vid_fps / reduction));
		int nsamples = v - u;
		std::vector<int16_t> samples(nsamples, 0);

		for (int k = 0; k < nsamples; k += 1)
			samples[k] = sin(audiofreq * 2 * M_PI / audiorate * (u + k))
			* ((1<<15)-1)
			* linmap(20 * log10(element.masksum), -90.0, 0.0, 0.0, 1.0);

		oaudio.write((char*)samples.data(), sizeof(samples[0]) * nsamples);

	}

	odiff.release();
	oaudio.close();

	std::cout << "writer is done\n";
}

std::vector<unsigned> histogram(cv::Mat image, double xmin, double xmax, int nbins)
{
	std::vector<unsigned> bins(nbins, 0);
	double const dx = (xmax - xmin) / nbins;

	for (int y = 0; y < image.rows; y += 1)
	for (int x = 0; x < image.cols; x += 1)
	{
		double const value = image.at<double>(y, x);
		int bin = (int)((value - xmin) / (xmax - xmin) * nbins);

		if (bin < 0) bin = 0;
		else if (bin >= nbins) bin = nbins-1;

		bins[bin] += 1;
	}

	return bins;
}

class ValuePlot
{
public:
	std::string windowname;
	cv::Mat plotdata;
	int xsize, ysize;

	float xmin, xmax, xstep;
	float vmin, vmax, vstep;

	bool use_vmap;
	std::function< float(float) > vmap;

public:
	//ValuePlot() = delete;

	ValuePlot(std::string windowname, int xsize, int ysize, float xmin=0, float xmax=10, float vmin=0, float vmax=10, float vstep=1) :
		windowname(windowname),
		xsize(xsize), ysize(ysize),
		xmin(xmin), xmax(xmax), xstep(1.0),
		vmin(vmin), vmax(vmax), vstep(vstep),
		use_vmap(false),
		vmap( [](float v) { return v; } )
	{
		//cv::namedWindow(windowname);
	}

	~ValuePlot()
	{
		//cv::destroyWindow(windowname);
	}

	void plot(cv::Mat data)
	{
		if (data.type() != CV_32F)
			return;

		if (use_vmap)
		{
			data = data.clone();
			for (int y = 0; y < data.rows; y += 1)
			{
				for (int x = 0; x < data.cols; x += 1)
				{
					data.at<float>(y,x) = vmap(data.at<float>(y,x));
				}
			}
		}

		//data = (data - ymax) * (ysize / (ymin-ymax)) + 0.5;
		double const alpha = (ysize-1) / (vmin-vmax);
		double const beta = (-vmax * (ysize-1)) / (vmin-vmax) + 0.5;
		
		data.convertTo(data, CV_32S, alpha, beta);

		plotdata.create(ysize, xsize, CV_32F);
		plotdata = 0.0;

		for (int row = 0; row < data.rows; row += 1)
		{
			for (int col = 0; col < data.cols; col += 1)
			{
				int xx = (int)(col * ((float)(xsize-1) / data.cols) + 0.5);
				if (xx < 0) xx = 0;
				else if (xx > xsize-1) xx = xsize-1;

				int yy = data.at<int32_t>(row,col);
				if (yy < 0) yy = 0;
				else if (yy > ysize-1) yy = ysize-1;

				plotdata.at<float>(yy,xx) += 1;
			}
		}

		plotdata /= (5 + plotdata);

		cv::cvtColor(plotdata, plotdata, CV_GRAY2BGR);

		//*
		std::ostringstream label;
		for (int k = floor(vmin/vstep); k <= vmax/vstep; k += 1)
		{
			float const v = k * vstep;
			label.str("");
			label << std::setprecision(2) << v;

			cv::Size textsize = cv::getTextSize(label.str(), CV_FONT_HERSHEY_PLAIN, 1.5, 2, nullptr);
			cv::line(plotdata, cv::Point(0, v*alpha + beta), cv::Point(xsize, v*alpha+beta), cv::Scalar(1.0,0.5,0.5));
			cv::putText(plotdata, label.str(), cv::Point(xsize - textsize.width, v*alpha + beta + textsize.height/2), CV_FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(1,1,1), 2);
		}
		//*/

		cv::imshow(windowname, plotdata);
	}
};

// TODO: plotting procedures for point clouds, histograms, ...
//       * grid, auto ticks
//       * log scales
//       * auto scaled axes
//       * map<x,y>, vector<y>, vector<x> + vector<y>

int main(int argc, char **argv)
{
	/*
	cv::Mat foo(50,1, CV_32FC3);
	cv::Mat bar(3,1, CV_32FC1);

	cv::Mat baz = bar.t() * foo.reshape(1).t();
	cout << baz.size() << endl;

	return 0;
	//*/

	parse_args(argc, argv);

	if (videofile.empty())
	{
		std::cout << "no input file given" << std::endl;
		return 0;
	}

	auto vid = new cv::VideoCapture(videofile);

	if (!vid->isOpened())
	{
		std::cerr << "vid could not be opened!" << std::endl;
		delete vid;
		exit(1);
	}

	// build file names
	char buf[1000];
	std::string basename = splitext(videofile);
	sprintf_s(buf, "%s-foreground-r%d%s-b%dx%d-a%g-d%g-s%g-mst%g", 
		basename.c_str(),
		reduction, (blendframes ? "b" : ""),
		blur.x, blur.y,
		alpha_avg,
		alpha_dev,
		sigma_dev,
		masksum_threshold
	);
	std::string outbase(buf);
	if (suffix.length() > 0) outbase += "-" + suffix;

	std::cout << "outbase: " << outbase << std::endl;

	std::string inmaskname(basename + "-inmask.bmp");
	cv::Mat inmask = cv::imread(inmaskname, cv::IMREAD_GRAYSCALE);
	bool const use_inmask = (inmask.data != NULL);
	if (use_inmask)
	{
		std::cout << "using inmask " << inmaskname << std::endl;
		//cv::imshow("inmask", inmask);
	}

	std::string outaudio(outbase + "-audio.raw"); // raw pcm_s16le
	//std::string outmask    (outbase + "-mask.avi");
	std::string outdiff    (outbase + "-maskdiff.avi");

	vid_fps = vid->get(CV_CAP_PROP_FPS);
	width = (int)vid->get(CV_CAP_PROP_FRAME_WIDTH);
	height = (int)vid->get(CV_CAP_PROP_FRAME_HEIGHT);
	nframes = (int)vid->get(CV_CAP_PROP_FRAME_COUNT);

	if (override_fps)
	{
		cout << "fps " << vid_fps << " uncorrected" << endl;
		cout << "nframes " << nframes << " uncorrected" << endl;

		// compensate
		nframes = nframes / vid_fps * override_fps;

		vid_fps = override_fps;
	}
	
	cout << "fps " << vid_fps << endl;
	cout << "nframes " << nframes << endl;

	assert(fmod((double)audiorate, vid_fps) < 1.0);

	cv::Mat frameu8(height, width, CV_8UC3, cv::Scalar::all(0.0f));
	cv::Mat frame(height, width, CV_32FC3, cv::Scalar::all(0.0f));
	cv::Mat gray(height, width, CV_32F, cv::Scalar::all(0.0f));
	cv::Mat average(height, width, CV_32F, cv::Scalar::all(0.0f));
	cv::Mat deviation(height, width, CV_32F, cv::Scalar::all(0.0f));

	ValuePlot plot_avg("plot avg", 640, 360, 0, 640, 0, 1, 0.2);
	
	ValuePlot plot_diff("plot diff", 640, 360, 0, 640, -0.03, 0.03, 1/256.);

	ValuePlot plot_absdiff("plot absdiff", 640, 360, 0, 640, -4, 0, 1);
	plot_absdiff.use_vmap = true;
	plot_absdiff.vmap = [](float v) { return log10(v); };
	
	ValuePlot plot_dev("plot dev", 640, 360, 0, 640, -4, 0);
	plot_dev.use_vmap = true;
	plot_dev.vmap = [](float v) { return log10(v); };

	cv::Mat diff(height, width, CV_32F, cv::Scalar::all(0.0f));
	cv::Mat absdiff(height, width, CV_32F, cv::Scalar::all(0.0f));
	cv::Mat viewdiff(height, width, CV_32F, cv::Scalar::all(0.0f));
	cv::Mat mask(height, width, CV_8U, cv::Scalar::all(0));

	std::thread reader(readfn, vid);
	std::thread writer(writefn, outaudio, outdiff);

#ifdef WIN32
	SetThreadPriority(writer.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif

	double sched = hrtimer();
	double dsched = 0.25;

	double statsched = hrtimer();
	double dstat = 1.0;
	double encode_fps = 0.0;
	double encode_alpha = 0.1;
	unsigned lastcount = 0, dcount = 0;

	// DEBUG
	//dstat = 0.0;

	for (int k = 0; k < 5; k += 1)
		donequeue.push(true);

	while (running)
	{
		MatOrNothing item;
		bool success = framequeue.try_pop(item);
		if (!success)
			continue;
		if (!item.anything)
			break;

		// FIXME: compensate for sporadic *fast* whole-image luminance fluctuations (not caught by sigma)
		
		donequeue.push(true);

		frame = item.frame;
		gray = item.gray;
		unsigned frameno = item.frameno;

		if (use_inmask)
			cv::subtract(gray, average, diff, inmask);
		else
			cv::subtract(gray, average, diff);

		cv::add(diff, 0.5, viewdiff);
		absdiff = cv::abs(diff);
		auto diffsum = (cv::sum(absdiff)[0] / (width * height));

		// =====================================================================
		// TODO: histogram: deviation from 'average', bin size = 0.1, -10..+10, y-log

		/*
		a_avg.push_back(cv::sum(diff)[0] / (double)(width * height));
		a_dev.push_back(cv::sum(deviation)[0] / (width * height));
		while (a_avg.size() > a_depth) a_avg.erase(a_avg.begin());
		while (a_dev.size() > a_depth) a_dev.erase(a_dev.begin());

		int histwidth = 500;
		double const emax = 1;
		double const emin = -2;

		a_dev.clear();
		a_dev.resize(histwidth+1, 0);

		for (int y = 0; y < diff.rows; y += 1)
		for (int x = 0; x < diff.cols; x += 1)
		{
			int const bin = 100 + (int)(deviation.at<float>(y,x) / 0.001);

			if (bin < 0) continue;
			if (bin >= a_dev.size()) continue;

			a_dev[bin] += 1;
		}

		for (int k = 0; k < a_dev.size(); k += 1)
			a_dev[k] = a_dev[k] ? log10(a_dev[k]) : -1;

		//*/
		// =====================================================================

		cv::compare(1 / sigma_dev * absdiff, deviation, mask, CV_CMP_GT);
		auto masksum = cv::countNonZero(mask) / (double)(width * height);

		if (masksum > masksum_threshold)
			mask = 0;

		//cv::Mat const mix[] = { mask, mask, mask };
		//cv::Mat mask3;
		//cv::merge(mix, 3, mask3);

		cv::Mat maskviewdiff(height, width, CV_32F, cv::Scalar::all(0.0f));
		maskviewdiff.setTo(cv::Scalar::all(0.5f));
		viewdiff.copyTo(maskviewdiff, mask);

		// =====================================================================
		/*
		{
			cv::Mat foo;

			foo = 1 - deviation / 0.005;
			cv::cvtColor(foo, foo, CV_GRAY2BGR);
			cv::resize(foo, foo, preview_size, 0, 0, CV_INTER_LINEAR);
			
			for (double a = -0.1; a <= 0.1; a += 0.01)
				cv::line(foo, cv::Point(100 + a / 0.001, 200 - 10), cv::Point(100 + a / 0.001, 200 + 10), cv::Scalar::all(0.5), 1);
			
			for (int k = 1; k < a_dev.size(); k += 1)
				cv::line(foo, cv::Point(k - 1, 200 - a_dev[k - 1] / 0.1), cv::Point(k, 200 - a_dev[k] / 0.1), cv::Scalar::all(1), 1);

			for (int k = 1; k < a_avg.size(); k += 1)
				cv::line(foo, cv::Point(k - 1, 200 - a_avg[k - 1] / 0.0001), cv::Point(k, 200 - a_avg[k] / 0.0001), cv::Scalar(1, 0, 0), 1);
			cv::imshow("debug dev", foo);

			foo = 1 - absdiff / 0.005;
			cv::resize(foo, foo, preview_size, 0, 0, CV_INTER_LINEAR);
			cv::imshow("debug diff", foo);
		}
		//*/
		// =====================================================================


		WriteElement const tmp = { false, frameno, maskviewdiff, masksum };
		writequeue.push(tmp);

		if (!headless && sched < hrtimer())
		{
			sched += dsched;

			cv::Mat tmp;

			//cv::Mat display = mask3.clone();
			mask.convertTo(tmp, CV_32F, 1/255.);
			std::ostringstream message;
			message << to_hms(frameno / vid_fps);
			message << ", #" << frameno << " @ " << vid_fps << " fps";
			message << ", mask " << std::fixed << std::setprecision(6) << masksum;
			message << ", diff " << std::fixed << std::setprecision(6) << diffsum;

			cv::putText(tmp, message.str(), cv::Point(5, height - 5), CV_FONT_HERSHEY_PLAIN, width / 600, cv::Scalar::all(1.0), width / 600);
			cv::resize(tmp, tmp, preview_size, 0, 0, CV_INTER_AREA);
			cv::imshow("mask", tmp);

			cv::resize(average, tmp, preview_size, 0, 0, CV_INTER_AREA);
			cv::imshow("average", tmp);

			cv::log(deviation, tmp);
			tmp = tmp * (0.3 / log(10)) + 1.0;
			cv::resize(tmp, tmp, preview_size, 0, 0, CV_INTER_AREA);
			cv::imshow("deviation", tmp);

			/*
			//plot_avg.plot(average);
			plot_absdiff.plot(absdiff);
			plot_diff.plot(diff);
			plot_dev.plot(deviation);
			//*/

			tmp = ((viewdiff - 0.5) * 5) + 0.5;
			cv::resize(tmp, tmp, preview_size, 0, 0, CV_INTER_AREA);
			cv::imshow("diff", tmp);

			cv::resize(maskviewdiff, tmp, preview_size, 0, 0, CV_INTER_AREA);
			cv::imshow("maskdiff", tmp);

			while (true)
			{
				int key = cv::waitKey(1);

				if (key == -1) break;
				if (key == 27) goto STOP;

				std::cout << "key " << key << " pressed" << std::endl;
			}
		}

		// updates
		cv::scaleAdd(absdiff - deviation, alpha_dev, deviation, deviation);

		cv::scaleAdd(diff, alpha_avg, average, average);

		if (statsched < hrtimer())
		{
			statsched += dstat;

			dcount = frameno - lastcount;
			lastcount = frameno;
			encode_fps = encode_alpha * dcount + (1 - encode_alpha) * encode_fps;

			double timeleft = (nframes - frameno) / encode_fps;

			std::cout << "frame " << frameno;
			std::cout << ", " << std::fixed << std::setprecision(2) << encode_fps << " fps";
			std::cout << ", " << std::fixed << std::setprecision(3) << (100. * frameno / nframes) << "%";
			std::cout << ", ETA " << std::fixed << std::setprecision(2) << (timeleft / 60) << "min  \r";
			std::cout.flush();
		}

	}
STOP:

	running = false;
	WriteElement const tmp = { true };
	writequeue.push(tmp);

	std::cout << std::endl << "done" << std::endl;

	if (reader.joinable())
		reader.join();

	if (writer.joinable())
		writer.join();

	vid->release();
	delete vid;

	//odiff.release();
	//omask.release();
	//oaudio.close();


	return 0;
}
