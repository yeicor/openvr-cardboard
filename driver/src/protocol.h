#pragma once

// Wire protocol shared with desktop/openvrcardboard/protocol.py and the Android app.
// See docs/PROTOCOL.md. All fields are little-endian.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ovrc::proto {

constexpr std::array<char, 4> kMagic{ 'O', 'V', 'R', 'C' };
constexpr uint8_t kVersion = 1;

enum class PacketType : uint8_t {
	State = 0x01,
	Haptic = 0x02,
	Status = 0x03,
	Imu = 0x20,
	TimePing = 0x21,
	TimePong = 0x22,
};

enum class DeviceId : uint8_t {
	Hmd = 0,
	LeftController = 1,
	RightController = 2,
	TrackerBase = 100,
};

enum DeviceFlags : uint8_t {
	PoseValid = 1u << 0,
	Connected = 1u << 1,
	OrientationOnly = 1u << 2,
};

enum Buttons : uint32_t {
	ButtonTriggerClick = 1u << 0,
	ButtonGripClick = 1u << 1,
	ButtonApplicationMenu = 1u << 2,
	ButtonSystem = 1u << 3,
	ButtonA = 1u << 4,
	ButtonB = 1u << 5,
	ButtonTrackpadClick = 1u << 6,
	ButtonJoystickClick = 1u << 7,
	ButtonTrackpadTouch = 1u << 8,
	ButtonTriggerTouch = 1u << 9,
	ButtonJoystickTouch = 1u << 10,
	ButtonGripTouch = 1u << 11,
	ButtonX = 1u << 12,
	ButtonY = 1u << 13,
	ButtonDpadUp = 1u << 14,
	ButtonDpadDown = 1u << 15,
	ButtonDpadLeft = 1u << 16,
	ButtonDpadRight = 1u << 17,
	ButtonSelect = 1u << 18,
	ButtonStart = 1u << 19,
};

constexpr std::size_t kMaxDevices = 16;
constexpr std::size_t kMaxInputs = 8;
constexpr std::size_t kMaxTrackers = 16;

#pragma pack( push, 1 )

struct Header {
	std::array<char, 4> magic;
	uint8_t version;
	uint8_t type;
	uint16_t flags;
	uint32_t sequence;
};

struct DevicePose {
	uint8_t id;
	uint8_t flags;
	uint16_t reserved;
	float position[ 3 ];
	float rotation[ 4 ]; // x, y, z, w
	float velocity[ 3 ];
	float angularVelocity[ 3 ];
};

struct ControllerInput {
	uint8_t id;
	uint8_t reserved[ 3 ];
	uint32_t buttons;
	float trigger;
	float grip;
	float joystickX;
	float joystickY;
	float trackpadX;
	float trackpadY;
	float reservedFloat;
};

// Receiving-side convenience type. The wire representation is variable length;
// ParseState() fills the first deviceCount/inputCount entries and zeroes the rest.
struct State {
	Header header;
	double timestamp;
	uint8_t deviceCount;
	uint8_t reserved0[ 3 ];
	// Horizon the sender already applied, in milliseconds. The driver maps it
	// to poseTimeOffset and never predicts a second time.
	float predictionMs;
	DevicePose devices[ kMaxDevices ];
	uint8_t inputCount;
	uint8_t reserved1[ 3 ];
	ControllerInput inputs[ kMaxInputs ];
};

struct Haptic {
	Header header;
	uint8_t deviceId;
	uint8_t reserved[ 3 ];
	float amplitude;
	float frequency;
	float durationSeconds;
};

struct Status {
	Header header;
	uint32_t protocolVersion;
	uint32_t deviceCount;
	uint32_t uptimeMs;
	uint32_t statePackets;
	uint32_t parseErrors;
};

struct Imu {
	Header header;
	double timestamp;
	float quaternion[ 4 ];
	float gyroscope[ 3 ];
	float acceleration[ 3 ];
	uint8_t accuracy;
	uint8_t reserved[ 3 ];
};

struct TimePing {
	Header header;
	double clientTime;
};

struct TimePong {
	Header header;
	double clientTime;
	double serverTime;
};

#pragma pack( pop )

static_assert( sizeof( Header ) == 12, "unexpected header size" );
static_assert( sizeof( DevicePose ) == 56, "unexpected DevicePose size" );
static_assert( sizeof( ControllerInput ) == 36, "unexpected ControllerInput size" );
static_assert( sizeof( Haptic ) == 28, "unexpected Haptic size" );
static_assert( sizeof( Status ) == 32, "unexpected Status size" );
static_assert( sizeof( Imu ) == 64, "unexpected Imu size" );

inline bool IsLittleEndian()
{
	const uint16_t value = 0x0001;
	return *reinterpret_cast<const uint8_t *>( &value ) == 1;
}

inline bool IsFinite( float value )
{
	return std::isfinite( value ) != 0;
}

inline bool IsFinite3( const float *v )
{
	return IsFinite( v[ 0 ] ) && IsFinite( v[ 1 ] ) && IsFinite( v[ 2 ] );
}

inline bool IsFinite4( const float *v )
{
	return IsFinite3( v ) && IsFinite( v[ 3 ] );
}

inline bool HasMagic( const Header &header )
{
	return std::memcmp( header.magic.data(), kMagic.data(), kMagic.size() ) == 0;
}

inline Header MakeHeader( PacketType type, uint32_t sequence )
{
	Header header{};
	header.magic = kMagic;
	header.version = kVersion;
	header.type = static_cast<uint8_t>( type );
	header.flags = 0;
	header.sequence = sequence;
	return header;
}

inline bool ParseHeader( const uint8_t *data, std::size_t size, Header &out )
{
	if ( data == nullptr || size < sizeof( Header ) )
		return false;
	std::memcpy( &out, data, sizeof( Header ) );
	if ( !HasMagic( out ) || out.version != kVersion )
		return false;
	return true;
}

inline bool ParseState( const uint8_t *data, std::size_t size, State &out, std::string *error = nullptr )
{
	out = State{};
	Header header{};
	if ( !ParseHeader( data, size, header ) || header.type != static_cast<uint8_t>( PacketType::State ) ) {
		if ( error )
			*error = "bad header";
		return false;
	}

	std::size_t offset = sizeof( Header );
	const std::size_t fixed = offset + sizeof( double ) + 1 + 3 + sizeof( float );
	if ( size < fixed ) {
		if ( error )
			*error = "truncated state prefix";
		return false;
	}

	std::memcpy( &out.timestamp, data + offset, sizeof( double ) );
	offset += sizeof( double );
	const uint8_t deviceCount = data[ offset ];
	offset += 1 + 3; // count + reserved
	std::memcpy( &out.predictionMs, data + offset, sizeof( float ) );
	offset += sizeof( float );
	if ( !std::isfinite( out.predictionMs ) || out.predictionMs < 0.0f )
		out.predictionMs = 0.0f;
	if ( out.predictionMs > 1000.0f )
		out.predictionMs = 1000.0f;
	if ( deviceCount > kMaxDevices ) {
		if ( error )
			*error = "too many devices";
		return false;
	}
	if ( size < offset + deviceCount * sizeof( DevicePose ) + 4 ) {
		if ( error )
			*error = "truncated device array";
		return false;
	}

	for ( uint8_t i = 0; i < deviceCount; ++i ) {
		std::memcpy( &out.devices[ i ], data + offset, sizeof( DevicePose ) );
		offset += sizeof( DevicePose );
	}
	const uint8_t inputCount = data[ offset ];
	offset += 1 + 3;
	if ( inputCount > kMaxInputs ) {
		if ( error )
			*error = "too many inputs";
		return false;
	}
	if ( size < offset + inputCount * sizeof( ControllerInput ) ) {
		if ( error )
			*error = "truncated input array";
		return false;
	}
	for ( uint8_t i = 0; i < inputCount; ++i ) {
		std::memcpy( &out.inputs[ i ], data + offset, sizeof( ControllerInput ) );
		offset += sizeof( ControllerInput );
	}

	if ( !std::isfinite( out.timestamp ) ) {
		if ( error )
			*error = "non-finite timestamp";
		return false;
	}

	out.header = header;
	out.deviceCount = deviceCount;
	out.inputCount = inputCount;
	return true;
}

inline bool ParseImu( const uint8_t *data, std::size_t size, Imu &out )
{
	if ( size < sizeof( Imu ) )
		return false;
	std::memcpy( &out, data, sizeof( Imu ) );
	if ( !HasMagic( out.header ) || out.header.version != kVersion ||
		 out.header.type != static_cast<uint8_t>( PacketType::Imu ) )
		return false;
	return std::isfinite( out.timestamp );
}

inline bool ParseTimePing( const uint8_t *data, std::size_t size, TimePing &out )
{
	if ( size < sizeof( TimePing ) )
		return false;
	std::memcpy( &out, data, sizeof( TimePing ) );
	return HasMagic( out.header ) && out.header.version == kVersion &&
		   out.header.type == static_cast<uint8_t>( PacketType::TimePing );
}

inline Haptic MakeHaptic( uint32_t sequence, uint8_t deviceId, float amplitude, float frequency, float durationSeconds )
{
	Haptic packet{};
	packet.header = MakeHeader( PacketType::Haptic, sequence );
	packet.deviceId = deviceId;
	packet.amplitude = amplitude;
	packet.frequency = frequency;
	packet.durationSeconds = durationSeconds;
	return packet;
}

inline Status MakeStatus( uint32_t sequence, uint32_t deviceCount, uint32_t uptimeMs, uint32_t statePackets, uint32_t parseErrors )
{
	Status packet{};
	packet.header = MakeHeader( PacketType::Status, sequence );
	packet.protocolVersion = kVersion;
	packet.deviceCount = deviceCount;
	packet.uptimeMs = uptimeMs;
	packet.statePackets = statePackets;
	packet.parseErrors = parseErrors;
	return packet;
}

inline TimePong MakeTimePong( uint32_t sequence, double clientTime, double serverTime )
{
	TimePong packet{};
	packet.header = MakeHeader( PacketType::TimePong, sequence );
	packet.clientTime = clientTime;
	packet.serverTime = serverTime;
	return packet;
}

} // namespace ovrc::proto
