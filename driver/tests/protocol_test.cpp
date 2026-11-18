#include <cmath>
#include <cstdio>
#include <vector>

#include "protocol.h"

using namespace ovrc;

namespace {

std::vector<uint8_t> BuildStatePacket()
{
	// Build the wire layout manually so this test does not simply mirror the
	// receiving code's struct layout assumptions.
	std::vector<uint8_t> bytes;
	auto append = [ & ]( const void *data, std::size_t size ) {
		const auto *start = static_cast<const uint8_t *>( data );
		bytes.insert( bytes.end(), start, start + size );
	};

	proto::Header header = proto::MakeHeader( proto::PacketType::State, 42 );
	append( &header, sizeof( header ) );
	const double timestamp = 12.5;
	append( &timestamp, sizeof( timestamp ) );
	const uint8_t deviceCount = 2;
	const uint8_t reserved[ 3 ] = { 0, 0, 0 };
	append( &deviceCount, 1 );
	append( reserved, 3 );
	const float predictionMs = 35.0f;
	append( &predictionMs, sizeof( predictionMs ) );

	proto::DevicePose hmd{};
	hmd.id = 0;
	hmd.flags = proto::PoseValid | proto::Connected;
	hmd.position[ 0 ] = 0.1f;
	hmd.position[ 1 ] = 1.7f;
	hmd.position[ 2 ] = -0.2f;
	hmd.rotation[ 3 ] = 1.0f;
	append( &hmd, sizeof( hmd ) );

	proto::DevicePose left{};
	left.id = 1;
	left.flags = proto::PoseValid | proto::Connected;
	left.position[ 0 ] = -0.3f;
	left.rotation[ 3 ] = 1.0f;
	append( &left, sizeof( left ) );

	const uint8_t inputCount = 1;
	append( &inputCount, 1 );
	append( reserved, 3 );
	proto::ControllerInput input{};
	input.id = 1;
	input.buttons = proto::ButtonTriggerClick | proto::ButtonA;
	input.trigger = 0.75f;
	input.grip = 0.25f;
	input.joystickX = -0.5f;
	append( &input, sizeof( input ) );
	return bytes;
}

int failures = 0;

#define CHECK( expression ) \
	do { \
		if ( !( expression ) ) { \
			std::fprintf( stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression ); \
			++failures; \
		} \
	} while ( 0 )

} // namespace

int main()
{
	CHECK( proto::IsLittleEndian() );
	CHECK( sizeof( proto::Header ) == 12 );
	CHECK( sizeof( proto::DevicePose ) == 56 );
	CHECK( sizeof( proto::ControllerInput ) == 36 );
	CHECK( sizeof( proto::Imu ) == 64 );
	CHECK( sizeof( proto::Status ) == 32 );
	CHECK( sizeof( proto::Haptic ) == 28 );

	const auto bytes = BuildStatePacket();
	proto::State state{};
	std::string error;
	CHECK( proto::ParseState( bytes.data(), bytes.size(), state, &error ) );
	CHECK( state.deviceCount == 2 );
	CHECK( state.inputCount == 1 );
	CHECK( state.header.sequence == 42 );
	CHECK( std::fabs( state.timestamp - 12.5 ) < 1e-9 );
	CHECK( std::fabs( state.predictionMs - 35.0f ) < 1e-6f );
	CHECK( state.devices[ 0 ].id == 0 );
	CHECK( std::fabs( state.devices[ 0 ].position[ 1 ] - 1.7f ) < 1e-6f );
	CHECK( state.devices[ 1 ].id == 1 );
	CHECK( state.inputs[ 0 ].buttons == ( proto::ButtonTriggerClick | proto::ButtonA ) );
	CHECK( std::fabs( state.inputs[ 0 ].trigger - 0.75f ) < 1e-6f );

	proto::State truncated{};
	CHECK( !proto::ParseState( bytes.data(), bytes.size() - 4, truncated ) );

	auto corrupt = bytes;
	corrupt[ 0 ] = 'X';
	CHECK( !proto::ParseState( corrupt.data(), corrupt.size(), truncated ) );

	auto noInputs = bytes;
	// Truncate just before the input count byte: device array is complete but
	// the trailing input section is missing.
	const std::size_t inputPrefixOffset = sizeof( proto::Header ) + sizeof( double ) + 1 + 3 +
		sizeof( float ) + ( 2 * sizeof( proto::DevicePose ) );
	CHECK( !proto::ParseState( noInputs.data(), inputPrefixOffset, truncated ) );

	const auto haptic = proto::MakeHaptic( 7, 1, 0.8f, 120.0f, 0.05f );
	CHECK( haptic.deviceId == 1 );
	CHECK( haptic.header.type == static_cast<uint8_t>( proto::PacketType::Haptic ) );
	CHECK( std::fabs( haptic.durationSeconds - 0.05f ) < 1e-6f );

	const auto status = proto::MakeStatus( 8, 5, 1000, 321, 0 );
	CHECK( status.deviceCount == 5 );
	CHECK( status.statePackets == 321 );

	if ( failures != 0 ) {
		std::fprintf( stderr, "%d protocol self-test failure(s)\n", failures );
		return 1;
	}
	std::printf( "protocol self-test passed\n" );
	return 0;
}
