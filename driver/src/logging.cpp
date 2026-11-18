#include "logging.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

#include "openvr_driver.h"

namespace ovrc {

void Log( const char *format, ... )
{
	if ( format == nullptr )
		return;
	char buffer[ 2048 ]{};
	va_list args;
	va_start( args, format );
	std::vsnprintf( buffer, sizeof( buffer ), format, args );
	va_end( args );

	static std::mutex mutex;
	std::lock_guard<std::mutex> lock( mutex );
	if ( vr::VRDriverLog() != nullptr )
		vr::VRDriverLog()->Log( buffer );
}

} // namespace ovrc
