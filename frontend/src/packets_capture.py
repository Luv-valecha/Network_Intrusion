from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import pandas as pd

def capture_packet_features(duration=10):
    packets = sniff(filter="ip", timeout=duration, store=True)
    
    connection_stats = defaultdict(lambda: {'count': 0, 'same_srv': 0, 'diff_srv': 0})
    host_stats = defaultdict(lambda: {'srv_count': 0, 'srv_serror_count': 0})
    traffic_stats = defaultdict(lambda: {'src_bytes': 0})  # for reverse tracking

    data = []

    for pkt in packets:
        if IP not in pkt:
            continue

        ip = pkt[IP]
        src_ip, dst_ip = ip.src, ip.dst
        proto = pkt.proto
        src_bytes = len(ip.payload)
        service = None
        flag = None
        serror = False

        if TCP in pkt:
            tcp = pkt[TCP]
            service = tcp.dport
            flag = str(tcp.flags)
            if 'R' in flag:
                serror = True

        elif UDP in pkt:
            udp = pkt[UDP]
            service = udp.dport
            flag = "U"
        
        else:
            continue  # skip non-TCP/UDP

        # Track forward and reverse traffic
        conn_key = (src_ip, dst_ip)
        reverse_key = (dst_ip, src_ip)
        traffic_stats[conn_key]['src_bytes'] += src_bytes
        dst_bytes = traffic_stats[reverse_key]['src_bytes']  # reverse flow

        # Update connection stats
        conn_service_key = (src_ip, dst_ip, service)
        connection_stats[conn_service_key]['count'] += 1
        host_stats[dst_ip]['srv_count'] += 1
        if serror:
            host_stats[dst_ip]['srv_serror_count'] += 1

        # Calculate same_srv and diff_srv
        same_srv = sum(1 for k in connection_stats if k[0] == src_ip and k[1] == dst_ip and k[2] == service)
        diff_srv = sum(1 for k in connection_stats if k[0] == src_ip and k[1] == dst_ip and k[2] != service)

        same_srv_rate = same_srv / max(1, connection_stats[conn_service_key]['count'])
        diff_srv_rate = diff_srv / max(1, connection_stats[conn_service_key]['count'])

        dst_host_srv_count = host_stats[dst_ip]['srv_count']
        dst_host_srv_serror_rate = host_stats[dst_ip]['srv_serror_count'] / max(1, dst_host_srv_count)

        features = {
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': same_srv_rate,  
            'dst_host_diff_srv_rate': diff_srv_rate,  
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,           
        }

        data.append(features)

    return pd.DataFrame(data)
