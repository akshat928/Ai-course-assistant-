# Unit 1: Computer Networks Fundamentals

## The OSI Model

The Open Systems Interconnection (OSI) model splits network communication into seven layers. From bottom to top they are: Physical, Data Link, Network, Transport, Session, Presentation and Application. Each layer serves the layer above it and relies on the layer below it.

The Physical layer transmits raw bits over a medium such as copper, fibre or radio. The Data Link layer groups bits into frames and handles MAC addressing and error detection within a single network segment; switches operate here. The Network layer routes packets between networks using logical addresses (IP addresses); routers operate here. The Transport layer provides end-to-end delivery between processes using port numbers.

The TCP/IP model used on the internet collapses these into four layers: Link, Internet, Transport and Application.

## TCP vs UDP

TCP (Transmission Control Protocol) is connection-oriented. Before data is sent, the client and server perform a three-way handshake: the client sends SYN, the server replies SYN-ACK, and the client answers ACK. TCP guarantees ordered, reliable delivery using sequence numbers, acknowledgements and retransmission of lost segments. It also performs flow control with a sliding window and congestion control with algorithms such as slow start.

UDP (User Datagram Protocol) is connectionless. It sends datagrams without a handshake, without acknowledgements and without ordering guarantees. Its header is only 8 bytes, compared with at least 20 bytes for TCP.

Use TCP when every byte must arrive correctly, such as web pages (HTTP), email (SMTP) and file transfer. Use UDP when low latency matters more than perfect delivery, such as live video, online games, VoIP and DNS lookups.

## IP Addressing and Subnetting

An IPv4 address is 32 bits, written as four decimal octets such as 192.168.1.10. A subnet mask separates the network part from the host part. In CIDR notation, 192.168.1.0/24 means the first 24 bits identify the network, leaving 8 bits for hosts.

The number of usable hosts in a subnet is 2^(host bits) minus 2, because the all-zeros address is the network address and the all-ones address is the broadcast address. A /24 subnet therefore has 254 usable hosts, and a /26 subnet has 62.

Private address ranges, which are not routed on the public internet, are 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16. Network Address Translation (NAT) lets many devices with private addresses share one public IP address.

## DNS

The Domain Name System translates human-readable names such as example.com into IP addresses. A lookup starts at a recursive resolver, which queries a root server, then the top-level domain server (for .com), then the authoritative name server for the domain. Results are cached according to their TTL (time to live) to reduce load and latency.

Common record types are A (IPv4 address), AAAA (IPv6 address), CNAME (alias to another name), MX (mail server) and TXT (arbitrary text, often used for domain verification and SPF email policies). DNS normally uses UDP port 53, falling back to TCP for large responses and zone transfers.
