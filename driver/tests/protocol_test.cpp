#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "protocol.h"

using namespace ovrc;

namespace {

std::unordered_map<std::string, std::vector<uint8_t>> LoadGoldenVectors()
{
	std::unordered_map<std::string, std::vector<uint8_t>> vectors;
	std::ifstream file( OVRC_GOLDEN_VECTORS );
	std::string line;
	while ( std::getline( file, line ) ) {
		if ( line.empty() || line[ 0 ] == '#' )
			continue;
		std::istringstream stream( line );
		std::string name;
		std::string hex;
		if ( !( stream >> name >> hex ) || hex.size() % 2 != 0 )
			continue;
		std::vector<uint8_t> bytes;
		bytes.reserve( hex.size() / 2 );
		for ( std::size_t i = 0; i < hex.size(); i += 2 )
			bytes.push_back( static_cast<uint8_t>( std::stoul( hex.substr( i, 2 ), nullptr, 16 ) ) );
		vectors.emplace( std::move( name ), std::move( bytes ) );
	}
	return vectors;
}

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
	left.position[ 1 ] = 1.2f;
	left.position[ 2 ] = -0.4f;
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
	input.joystickY = 0.5f;
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

	const auto golden = LoadGoldenVectors();
	auto goldenBytes = [ & ]( const char *name ) -> const std::vector<uint8_t> * {
		const auto it = golden.find( name );
		return it == golden.end() ? nullptr : &it->second;
	};
	const auto *goldenState = goldenBytes( "state" );
	const auto *goldenHaptic = goldenBytes( "haptic" );
	const auto *goldenStatus = goldenBytes( "status" );
	const auto *goldenImu = goldenBytes( "imu" );
	const auto *goldenImuAuth = goldenBytes( "imu_auth" );
	const auto *goldenPing = goldenBytes( "time_ping" );
	const auto *goldenPingAuth = goldenBytes( "time_ping_auth" );
	CHECK( goldenState != nullptr );
	CHECK( goldenHaptic != nullptr );
	CHECK( goldenStatus != nullptr );
	CHECK( goldenImu != nullptr );
	CHECK( goldenImuAuth != nullptr );
	CHECK( goldenPing != nullptr );
	CHECK( goldenPingAuth != nullptr );

	// C++ must produce exactly the same bytes as the Python and Kotlin
	// encoders, not merely parse its own layout.
	if ( goldenState != nullptr )
		CHECK( bytes == *goldenState );
	if ( goldenHaptic != nullptr ) {
		CHECK( goldenHaptic->size() == sizeof( proto::Haptic ) );
		CHECK( std::memcmp( &haptic, goldenHaptic->data(), sizeof( proto::Haptic ) ) == 0 );
	}
	if ( goldenStatus != nullptr ) {
		const auto goldenStatusPacket = proto::MakeStatus( 8, 5, 1000, 321, 2 );
		CHECK( goldenStatus->size() == sizeof( proto::Status ) );
		CHECK( std::memcmp( &goldenStatusPacket, goldenStatus->data(), sizeof( proto::Status ) ) == 0 );
	}
	if ( goldenImu != nullptr ) {
		proto::Imu imu{};
		CHECK( proto::ParseImu( goldenImu->data(), goldenImu->size(), imu ) );
		CHECK( std::fabs( imu.timestamp - 1234.5 ) < 1e-9 );
		CHECK( std::fabs( imu.quaternion[ 3 ] - 0.9f ) < 1e-6f );
	}
	if ( goldenImuAuth != nullptr ) {
		proto::Imu imu{};
		// The driver ignores the optional 16-byte authenticator on packets it
		// does not consume; accepting the longer form keeps the vector shared.
		CHECK( goldenImuAuth->size() == sizeof( proto::Imu ) + 16 );
		CHECK( proto::ParseImu( goldenImuAuth->data(), goldenImuAuth->size(), imu ) );
	}
	if ( goldenPing != nullptr ) {
		proto::TimePing ping{};
		CHECK( proto::ParseTimePing( goldenPing->data(), goldenPing->size(), ping ) );
		CHECK( std::fabs( ping.clientTime - 55.125 ) < 1e-9 );
	}
	if ( goldenPingAuth != nullptr ) {
		proto::TimePing ping{};
		CHECK( goldenPingAuth->size() == sizeof( proto::TimePing ) + 16 );
		CHECK( proto::ParseTimePing( goldenPingAuth->data(), goldenPingAuth->size(), ping ) );
	}

	if ( failures != 0 ) {
		std::fprintf( stderr, "%d protocol self-test failure(s)\n", failures );
		return 1;
	}
	std::printf( "protocol self-test passed\n" );
	return 0;
}
