#include "udp_socket.h"

#include <cstring>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <sys/select.h>
#include <unistd.h>
#endif

namespace ovrc {

namespace {

#ifdef _WIN32
void EnsureWinsock()
{
	static std::once_flag flag;
	std::call_once( flag, [] {
		WSADATA data{};
		WSAStartup( MAKEWORD( 2, 2 ), &data );
	} );
}
#else
void SetNonBlocking( int fd )
{
	const int flags = fcntl( fd, F_GETFL, 0 );
	if ( flags >= 0 )
		fcntl( fd, F_SETFL, flags | O_NONBLOCK );
}
#endif

void SetError( std::string *error, const std::string &message )
{
	if ( error )
		*error = message;
}

} // namespace

UdpSocket::~UdpSocket()
{
	Close();
}

bool UdpSocket::Bind( uint16_t port, bool loopbackOnly, std::string *error )
{
	Close();
#ifdef _WIN32
	EnsureWinsock();
	handle_ = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
	if ( handle_ == INVALID_SOCKET ) {
		SetError( error, "socket() failed" );
		return false;
	}
	BOOL reuse = TRUE;
	setsockopt( handle_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>( &reuse ), sizeof( reuse ) );
#else
	handle_ = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
	if ( handle_ < 0 ) {
		SetError( error, std::string( "socket() failed: " ) + std::strerror( errno ) );
		return false;
	}
	const int reuse = 1;
	setsockopt( handle_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof( reuse ) );
#endif

	sockaddr_in address{};
	address.sin_family = AF_INET;
	address.sin_port = htons( port );
	address.sin_addr.s_addr = htonl( loopbackOnly ? INADDR_LOOPBACK : INADDR_ANY );

	if ( bind( handle_, reinterpret_cast<sockaddr *>( &address ), sizeof( address ) ) != 0 ) {
#ifdef _WIN32
		SetError( error, "bind() failed: " + std::to_string( WSAGetLastError() ) );
#else
		SetError( error, std::string( "bind() failed: " ) + std::strerror( errno ) );
#endif
		Close();
		return false;
	}

#ifndef _WIN32
	SetNonBlocking( handle_ );
#endif
	return true;
}

void UdpSocket::Close()
{
#ifdef _WIN32
	if ( handle_ != INVALID_SOCKET ) {
		closesocket( handle_ );
		handle_ = INVALID_SOCKET;
	}
#else
	if ( handle_ >= 0 ) {
		close( handle_ );
		handle_ = -1;
	}
#endif
}

bool UdpSocket::IsOpen() const
{
#ifdef _WIN32
	return handle_ != INVALID_SOCKET;
#else
	return handle_ >= 0;
#endif
}

int UdpSocket::RecvFrom( void *buffer, std::size_t capacity, UdpEndpoint *from, int timeoutMs, std::string *error )
{
	if ( !IsOpen() )
		return -1;

	fd_set readSet;
	FD_ZERO( &readSet );
	FD_SET( handle_, &readSet );
	timeval timeout{};
	timeval *timeoutPtr = nullptr;
	if ( timeoutMs >= 0 ) {
		timeout.tv_sec = timeoutMs / 1000;
		timeout.tv_usec = ( timeoutMs % 1000 ) * 1000;
		timeoutPtr = &timeout;
	}

	const int ready = select( static_cast<int>( handle_ ) + 1, &readSet, nullptr, nullptr, timeoutPtr );
	if ( ready == 0 )
		return 0;
	if ( ready < 0 ) {
#ifdef _WIN32
		SetError( error, "select() failed: " + std::to_string( WSAGetLastError() ) );
#else
		if ( errno == EINTR )
			return 0;
		SetError( error, std::string( "select() failed: " ) + std::strerror( errno ) );
#endif
		return -1;
	}

	UdpEndpoint source{};
#ifdef _WIN32
	int sourceLength = sizeof( source.address );
	const int received = recvfrom( handle_, static_cast<char *>( buffer ), static_cast<int>( capacity ), 0,
		reinterpret_cast<sockaddr *>( &source.address ), &sourceLength );
	source.length = sourceLength;
	if ( received == SOCKET_ERROR ) {
		const int code = WSAGetLastError();
		if ( code == WSAEWOULDBLOCK )
			return 0;
		SetError( error, "recvfrom() failed: " + std::to_string( code ) );
		return -1;
	}
#else
	socklen_t sourceLength = sizeof( source.address );
	const ssize_t received = recvfrom( handle_, buffer, capacity, 0,
		reinterpret_cast<sockaddr *>( &source.address ), &sourceLength );
	source.length = sourceLength;
	if ( received < 0 ) {
		if ( errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR )
			return 0;
		SetError( error, std::string( "recvfrom() failed: " ) + std::strerror( errno ) );
		return -1;
	}
#endif
	if ( from )
		*from = source;
	return static_cast<int>( received );
}

bool UdpSocket::SendTo( const void *data, std::size_t size, const UdpEndpoint &to, std::string *error )
{
	if ( !IsOpen() )
		return false;
	const int sent = sendto( handle_, static_cast<const char *>( data ), static_cast<int>( size ), 0,
		reinterpret_cast<const sockaddr *>( &to.address ), to.length );
	if ( sent < 0 || static_cast<std::size_t>( sent ) != size ) {
#ifdef _WIN32
		SetError( error, "sendto() failed: " + std::to_string( WSAGetLastError() ) );
#else
		SetError( error, std::string( "sendto() failed: " ) + std::strerror( errno ) );
#endif
		return false;
	}
	return true;
}

std::string UdpSocket::EndpointToString( const UdpEndpoint &endpoint )
{
	char host[ INET6_ADDRSTRLEN ]{};
	uint16_t port = 0;
	if ( endpoint.address.ss_family == AF_INET ) {
		const auto *address = reinterpret_cast<const sockaddr_in *>( &endpoint.address );
		inet_ntop( AF_INET, &address->sin_addr, host, sizeof( host ) );
		port = ntohs( address->sin_port );
	} else if ( endpoint.address.ss_family == AF_INET6 ) {
		const auto *address = reinterpret_cast<const sockaddr_in6 *>( &endpoint.address );
		inet_ntop( AF_INET6, &address->sin6_addr, host, sizeof( host ) );
		port = ntohs( address->sin6_port );
	} else {
		return "unknown";
	}
	return std::string( host ) + ":" + std::to_string( port );
}

} // namespace ovrc
