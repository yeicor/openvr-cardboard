#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netinet/in.h>
#include <sys/socket.h>
#endif

namespace ovrc {

struct UdpEndpoint {
	sockaddr_storage address{};
#ifdef _WIN32
	int length = 0;
#else
	socklen_t length = 0;
#endif
};

// Small cross-platform UDP socket. One socket is used for both directions so a
// driver can answer to the source address of the latest state packet.
class UdpSocket {
public:
	UdpSocket() = default;
	~UdpSocket();
	UdpSocket( const UdpSocket & ) = delete;
	UdpSocket &operator=( const UdpSocket & ) = delete;

	bool Bind( uint16_t port, bool loopbackOnly, std::string *error );
	void Close();
	bool IsOpen() const;

	// Returns >0 bytes, 0 on timeout, -1 on error.
	int RecvFrom( void *buffer, std::size_t capacity, UdpEndpoint *from, int timeoutMs, std::string *error );
	bool SendTo( const void *data, std::size_t size, const UdpEndpoint &to, std::string *error );

	static std::string EndpointToString( const UdpEndpoint &endpoint );

private:
#ifdef _WIN32
	SOCKET handle_ = INVALID_SOCKET;
#else
	int handle_ = -1;
#endif
};

} // namespace ovrc
