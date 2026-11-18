#include "time_util.h"

#include <chrono>

namespace ovrc {

double MonotonicSeconds()
{
	using namespace std::chrono;
	static const steady_clock::time_point origin = steady_clock::now();
	return duration<double>( steady_clock::now() - origin ).count();
}

} // namespace ovrc
