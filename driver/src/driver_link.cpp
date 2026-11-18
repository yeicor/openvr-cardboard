#include "driver_link.h"

#include <algorithm>
#include <cstring>

#include "logging.h"
#include "time_util.h"

namespace ovrc {

DriverLink::~DriverLink()
{
	Stop();
}

bool DriverLink::Start( uint16_t port, StateCallback callback, std::string *error )
{
	Stop();
	callback_ = std::move( callback );
	port_ = port;
	hasSender_ = false;
	if ( !socket_.Bind( port, true, error ) ) {
		Log( "driver link: unable to bind UDP port %u: %s", (unsigned)port, error ? error->c_str() : "unknown" );
		return false;
	}
	running_ = true;
	thread_ = std::thread( &DriverLink::ReceiveLoop, this );
	Log( "driver link: listening on 127.0.0.1:%u", (unsigned)port );
	return true;
}

void DriverLink::Stop()
{
	if ( running_.exchange( false ) ) {
		if ( thread_.joinable() )
			thread_.join();
	}
	socket_.Close();
}

bool DriverLink::SendToClient( const void *data, std::size_t size )
{
	std::lock_guard<std::mutex> lock( senderMutex_ );
	if ( !hasSender_ )
		return false;
	std::string error;
	if ( !socket_.SendTo( data, size, lastSender_, &error ) ) {
		Log( "driver link: send failed: %s", error.c_str() );
		return false;
	}
	return true;
}

bool DriverLink::SendHaptic( uint8_t deviceId, float amplitude, float frequency, float durationSeconds )
{
	const auto packet = proto::MakeHaptic( hapticSequence_.fetch_add( 1 ), deviceId, amplitude, frequency, durationSeconds );
	return SendToClient( &packet, sizeof( packet ) );
}

bool DriverLink::SendStatus( uint32_t deviceCount )
{
	const auto packet = proto::MakeStatus( statusSequence_.fetch_add( 1 ), deviceCount,
		static_cast<uint32_t>( MonotonicSeconds() * 1000.0 ),
		static_cast<uint32_t>( statePackets_.load() ), parseErrors_.load() );
	return SendToClient( &packet, sizeof( packet ) );
}

bool DriverLink::HasClient() const
{
	std::lock_guard<std::mutex> lock( senderMutex_ );
	return hasSender_;
}

void DriverLink::ReceiveLoop()
{
	uint8_t buffer[ 4096 ];
	uint32_t loggedErrors = 0;

	while ( running_.load() ) {
		std::string error;
		UdpEndpoint sender{};
		const int received = socket_.RecvFrom( buffer, sizeof( buffer ), &sender, 200, &error );
		if ( received < 0 ) {
			Log( "driver link: receive error: %s", error.c_str() );
			continue;
		}
		if ( received == 0 )
			continue;

		proto::Header header{};
		if ( !proto::ParseHeader( buffer, static_cast<std::size_t>( received ), header ) )
			continue;

		switch ( static_cast<proto::PacketType>( header.type ) ) {
		case proto::PacketType::State: {
			proto::State state{};
			if ( !proto::ParseState( buffer, static_cast<std::size_t>( received ), state, &error ) ) {
				++parseErrors_;
				if ( loggedErrors++ < 10 )
					Log( "driver link: bad state packet: %s (%d bytes)", error.c_str(), received );
				break;
			}
			{
				std::lock_guard<std::mutex> lock( senderMutex_ );
				lastSender_ = sender;
				hasSender_ = true;
			}
			// Reject duplicates and packets that arrive out of order.  The
			// signed difference handles 32-bit sequence wraparound; a new
			// sender resets the comparison so a restarted desktop app is not
			// mistaken for a stale stream.
			const std::string senderKey = UdpSocket::EndpointToString( sender );
			const std::string previousKey = haveStateSequence_
				? UdpSocket::EndpointToString( lastStateSender_ )
				: std::string();
			if ( haveStateSequence_ && senderKey == previousKey ) {
				const int32_t delta = static_cast<int32_t>( header.sequence - lastStateSequence_ );
				if ( delta <= 0 ) {
					droppedStatePackets_.fetch_add( 1 );
					break;
				}
			}
			lastStateSender_ = sender;
			haveStateSequence_ = true;
			lastStateSequence_ = header.sequence;
			lastStateTime_ = MonotonicSeconds();
			statePackets_.fetch_add( 1 );
			if ( callback_ )
				callback_( state );
			break;
		}
		case proto::PacketType::TimePing: {
			proto::TimePing ping{};
			if ( !proto::ParseTimePing( buffer, static_cast<std::size_t>( received ), ping ) )
				break;
			const auto pong = proto::MakeTimePong( header.sequence, ping.clientTime, MonotonicSeconds() );
			std::string sendError;
			socket_.SendTo( &pong, sizeof( pong ), sender, &sendError );
			break;
		}
		default:
			break;
		}
	}
}

} // namespace ovrc
