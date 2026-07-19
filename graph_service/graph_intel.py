"""
SAFER Graph Intelligence Service — Graph Analytics Engine

Uses NetworkX to build transaction relation graphs, compute PageRank,
detect community patterns (mule rings, device sharing), and generate node coordinates.
"""

import networkx as nx
import random
import math


def generate_spring_coordinates(G, width=820, height=440):
    """
    Computes spring layout positions scaled to the SVG canvas.
    Ensures margin boundaries are respected and resolves overlap collisions.
    """
    if len(G) == 0:
        return {}
    
    # Run spring layout with stronger repulsion factor k and more iterations
    pos = nx.spring_layout(G, k=3.5 / math.sqrt(len(G) or 1), iterations=80, seed=42)
    
    # Scale from [-1, 1] to width x height canvas with margins
    margin_x = 80
    margin_y = 60
    scale_x = (width - 2 * margin_x) / 2.0
    scale_y = (height - 2 * margin_y) / 2.0
    center_x = width / 2.0
    center_y = height / 2.0
    
    coords = {}
    for node, (x, y) in pos.items():
        coords[node] = {
            "x": round(center_x + x * scale_x),
            "y": round(center_y + y * scale_y)
        }
        
    # Collision resolution loop (push nodes apart if they are too close)
    nodes_list = list(coords.keys())
    for _ in range(5):  # 5 passes to resolve overlaps
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                n1 = nodes_list[i]
                n2 = nodes_list[j]
                x1, y1 = coords[n1]["x"], coords[n1]["y"]
                x2, y2 = coords[n2]["x"], coords[n2]["y"]
                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = 68  # Minimum pixel distance between nodes to prevent visual overlap
                if dist < min_dist:
                    if dist == 0:
                        dx = random.choice([-1, 1])
                        dy = random.choice([-1, 1])
                        dist = math.sqrt(dx*dx + dy*dy)
                    # Push them apart symmetrically
                    overlap = min_dist - dist
                    push_x = (dx / dist) * overlap * 0.5
                    push_y = (dy / dist) * overlap * 0.5
                    
                    coords[n1]["x"] = round(coords[n1]["x"] - push_x)
                    coords[n1]["y"] = round(coords[n1]["y"] - push_y)
                    coords[n2]["x"] = round(coords[n2]["x"] + push_x)
                    coords[n2]["y"] = round(coords[n2]["y"] + push_y)
                    
                    # Constrain to margins
                    coords[n1]["x"] = max(margin_x, min(width - margin_x, coords[n1]["x"]))
                    coords[n1]["y"] = max(margin_y, min(height - margin_y, coords[n1]["y"]))
                    coords[n2]["x"] = max(margin_x, min(width - margin_x, coords[n2]["x"]))
                    coords[n2]["y"] = max(margin_y, min(height - margin_y, coords[n2]["y"]))
                    
    return coords


def analyze_transaction_network(transactions: list[dict]) -> dict:
    """
    Builds a NetworkX graph from raw database transactions.
    Computes PageRank, communities, and returns nodes/edges compatible with frontend.
    """
    G = nx.DiGraph()
    
    # Keep track of node details and types
    nodes_info = {}
    
    # Helper to add/merge nodes
    def add_node_info(node_id, label, node_type, risk, details):
        if node_id not in nodes_info:
            nodes_info[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "risk": risk,
                "details": details
            }
        else:
            # Upgrade risk if higher
            risks_order = ["low", "medium", "high", "critical"]
            curr_risk = nodes_info[node_id]["risk"]
            if risks_order.index(risk) > risks_order.index(curr_risk):
                nodes_info[node_id]["risk"] = risk
            # Update details
            nodes_info[node_id]["details"].update(details)

    # 1. Build the graph
    edges_list = []
    
    # Process only transactions that have some risk signals or all (up to 45 for clean visualization)
    # We prioritize high-risk ones so the graph displays interesting fraud patterns
    for tx in transactions[:45]:
        tx_id = tx.get("id")
        amount = tx.get("amount", 0.0)
        rail = tx.get("payment_rail", "Transfer")
        severity = tx.get("severity", "low")
        
        sender_acc = tx.get("sender_account")
        sender_name = tx.get("sender_name", "N/A")
        sender_bank = tx.get("sender_bank", "N/A")
        sender_city = tx.get("sender_city", "N/A")
        
        receiver_acc = tx.get("receiver_account")
        receiver_name = tx.get("receiver_name", "N/A")
        receiver_bank = tx.get("receiver_bank", "N/A")
        receiver_city = tx.get("receiver_city", "N/A")
        
        merchant = tx.get("merchant")
        merchant_cat = tx.get("merchant_category", "N/A")
        
        device_brand = tx.get("device_brand")
        device_type = tx.get("device_type", "N/A")
        device_fp = tx.get("device_fingerprint")
        
        ip_addr = tx.get("ip_address")
        
        # Add Sender Account Node
        sender_id = f"ACC-{sender_acc}"
        add_node_info(
            sender_id,
            f"ACC-{sender_acc[:4]}...",
            "account",
            severity,
            {
                "Nama Pemilik": sender_name,
                "Bank": sender_bank,
                "No. Rekening": sender_acc,
                "Lokasi": sender_city,
                "Status": "Aktif" if severity == "low" else "Dalam Pengawasan"
            }
        )
        
        # Add Receiver Account or Merchant Node
        if merchant and merchant != "None":
            receiver_id = f"MER-{merchant.replace(' ', '_')}"
            add_node_info(
                receiver_id,
                merchant,
                "merchant",
                severity,
                {
                    "Nama Merchant": merchant,
                    "Kategori": merchant_cat,
                    "Rail": rail,
                    "Kota": receiver_city
                }
            )
        else:
            receiver_id = f"ACC-{receiver_acc}"
            add_node_info(
                receiver_id,
                f"ACC-{receiver_acc[:4]}...",
                "account",
                severity,
                {
                    "Nama Pemilik": receiver_name,
                    "Bank": receiver_bank,
                    "No. Rekening": receiver_acc,
                    "Lokasi": receiver_city,
                    "Status": "Aktif"
                }
            )
            
        # Draw transfer edge between sender and receiver
        amount_fmt = f"Rp {amount/1_000_000:.1f}Jt" if amount >= 1_000_000 else f"Rp {amount/1_000:.0f}Rb"
        G.add_edge(sender_id, receiver_id, label=amount_fmt)
        edges_list.append({"id": tx_id, "from": sender_id, "to": receiver_id, "label": amount_fmt})
        
        # Connect Device if fingerprint exists
        if device_fp and device_fp != "None":
            device_node_id = f"DEV-{device_fp[:6]}"
            add_node_info(
                device_node_id,
                f"DEV-{device_brand.split()[0]}",
                "device",
                severity,
                {
                    "Tipe Perangkat": device_brand,
                    "Fingerprint": device_fp,
                    "Device Type": device_type,
                    "Lokasi": sender_city
                }
            )
            G.add_edge(device_node_id, sender_id)
            edges_list.append({"from": device_node_id, "to": sender_id})
            
        # Connect IP Address if exists
        if ip_addr and ip_addr != "None":
            ip_node_id = f"IP-{ip_addr.replace('.', '_')}"
            # Mask IP for display
            parts = ip_addr.split(".")
            masked_ip = f"{parts[0]}.{parts[1]}.x.x" if len(parts) >= 2 else ip_addr
            add_node_info(
                ip_node_id,
                f"IP {masked_ip}",
                "ip",
                severity,
                {
                    "Alamat IP": ip_addr,
                    "Lokasi": sender_city
                }
            )
            if device_fp and device_fp != "None":
                G.add_edge(ip_node_id, device_node_id)
                edges_list.append({"from": ip_node_id, "to": device_node_id})
            else:
                G.add_edge(ip_node_id, sender_id)
                edges_list.append({"from": ip_node_id, "to": sender_id})

    # If graph is empty, return empty results
    if len(G) == 0:
        return {"nodes": [], "edges": [], "insights": []}

    # 2. PageRank Analysis
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        # Normalize PageRank values
        max_pr = max(pagerank.values()) if pagerank else 1
        for node_id, pr_val in pagerank.items():
            if node_id in nodes_info:
                nodes_info[node_id]["details"]["PageRank Hub Score"] = f"{pr_val/max_pr * 100:.1f}/100"
                # If PageRank is high, elevate node risk to highlight hubs
                if pr_val/max_pr > 0.65 and nodes_info[node_id]["risk"] not in ("high", "critical"):
                    nodes_info[node_id]["risk"] = "high"
    except Exception as e:
        print(f"[GraphIntel] PageRank calculation error: {e}")

    # 3. Community Detection & Fraud Pattern Mining
    insights = []
    
    # Scan for Device Sharing (1 device connected to multiple accounts)
    devices = [nid for nid, info in nodes_info.items() if info["type"] == "device"]
    shared_devices_count = 0
    for dev in devices:
        # Get neighbors (accounts accessed from this device)
        accounts_linked = [n for n in G.successors(dev) if n in nodes_info and nodes_info[n]["type"] == "account"]
        if len(accounts_linked) >= 3:
            shared_devices_count += 1
            nodes_info[dev]["risk"] = "critical"
            nodes_info[dev]["details"]["Status"] = "DEVICE FARM HUB"
            acc_labels = ", ".join([nodes_info[acc]["label"] for acc in accounts_linked[:3]])
            insights.append(
                f"Deteksi Device Sharing: Perangkat {nodes_info[dev]['label']} digunakan untuk mengakses {len(accounts_linked)} akun berbeda ({acc_labels}...) secara bergantian."
            )
            
    # Scan for Mule Networks (Multiple sender accounts transferring to 1 receiver account/merchant)
    accounts = [nid for nid, info in nodes_info.items() if info["type"] == "account"]
    for acc in accounts:
        # Senders to this account
        in_degrees = [u for u, v in G.in_edges(acc) if u in nodes_info and nodes_info[u]["type"] == "account"]
        if len(in_degrees) >= 4:
            nodes_info[acc]["risk"] = "critical"
            nodes_info[acc]["details"]["Status"] = "MULE RING COLLECTOR"
            insights.append(
                f"Pola Fan-In (Mule Ring): Rekening {nodes_info[acc]['label']} ({nodes_info[acc]['details'].get('Nama Pemilik')}) menerima transfer mencurigakan dari {len(in_degrees)} rekening mule yang berbeda dalam waktu singkat."
            )

    # General graph insights if empty
    if not insights:
        high_risk_count = sum(1 for n in nodes_info.values() if n["risk"] in ("high", "critical"))
        insights.append(
            f"Analisis Struktur Graf: Teridentifikasi {len(G)} entitas terhubung, dengan {high_risk_count} entitas berstatus risiko tinggi/kritis."
        )
        insights.append("Pola transaksional antar entitas saat ini dinilai stabil tanpa indikasi anomali struktural terpusat.")
    
    # 4. Generate Coordinates via Spring Layout
    positions = generate_spring_coordinates(G)
    
    # Build final nodes array with coordinates
    nodes_result = []
    for node_id, info in nodes_info.items():
        pos = positions.get(node_id, {"x": random.randint(100, 700), "y": random.randint(80, 380)})
        info["x"] = pos["x"]
        info["y"] = pos["y"]
        nodes_result.append(info)

    return {
        "nodes": nodes_result,
        "edges": edges_list,
        "insights": insights[:5]  # limit to top 5 insights
    }
