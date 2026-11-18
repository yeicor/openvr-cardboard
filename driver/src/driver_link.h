#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "protocol.h"
#include "udp_socket.h"

namespace ovrc {

// UDP endpoint used by the desktop app to stream device state. The receiver
// runs on its own thread so poses can be applied at the sender's rate without
// waiting for SteamVR's RunFrame cadence.
class DriverLink {
public:
	using StateCallback = std::function<void( const proto::State & )>;

	~DriverLink();

	bool Start( uint16_t port, StateCallback callback, std::string *error );
	void Stop();

	bool SendHaptic( uint8_t deviceId, float amplitude, float frequency, float durationSeconds );
	bool SendStatus( uint32_t deviceCount );
	bool HasClient() const;

	uint64_t StatePacketCount() const { return statePackets_.load(); }
	uint32_t ParseErrorCount() const { return parseErrors_.load(); }
	uint32_t DroppedStateCount() const { return droppedStatePackets_.load(); }
	double LastStateTime() const { return lastStateTime_.load(); }
	uint16_t Port() const { return port_; }

private:
	void ReceiveLoop();
	bool SendToClient( const void *data, std::size_t size );

	UdpSocket socket_;
	std::thread thread_;
	std::atomic<bool> running_{ false };
	StateCallback callback_;
	uint16_t port_ = 0;

	mutable std::mutex senderMutex_;
	UdpEndpoint lastSender_{};
	bool hasSender_ = false;
	UdpEndpoint lastStateSender_{};
	bool haveStateSequence_ = false;
	uint32_t lastStateSequence_ = 0;

	std::atomic<uint64_t> statePackets_{ 0 };
	std::atomic<uint32_t> parseErrors_{ 0 };
	std::atomic<uint32_t> droppedStatePackets_{ 0 };
	std::atomic<double> lastStateTime_{ 0.0 };
	std::atomic<uint32_t> hapticSequence_{ 0 };
	std::atomic<uint32_t> statusSequence_{ 0 };
};

} // namespace ovrc
