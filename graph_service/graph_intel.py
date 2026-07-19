"""
SAFER Graph Intelligence Service — Graph Analytics Engine

Uses NetworkX to build transaction relation graphs, compute PageRank,
detect community patterns (mule rings, device sharing), and generate node coordinates.
"""

import networkx as nx
import random
import math


def generate_hierarchical_coordinates(G, nodes_info, width=820, height=440):
    """
    Computes hierarchical coordinates based on entity roles to prevent overlapping
    and display a clean cash-flow layout (IP -> Device -> Sender -> Receiver -> Merchant).
    """
    coords = {}
    
    # Categorize nodes into 5 tiers
    tier_nodes = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for node_id in G.nodes():
        info = nodes_info.get(node_id, {})
        ntype = info.get("type", "account")
        
        if ntype == "ip":
            tier_nodes[1].append(node_id)
        elif ntype == "device":
            tier_nodes[2].append(node_id)
        elif ntype == "merchant":
            tier_nodes[5].append(node_id)
        else: # account
            # Check in-degree to distinguish sender vs receiver
            in_deg = G.in_degree(node_id)
            if in_deg == 0:
                tier_nodes[3].append(node_id)
            else:
                tier_nodes[4].append(node_id)
                
    # Define vertical positions for each tier (IP -> Device -> Sender -> Receiver -> Merchant)
    tier_y = {
        1: 50,    # IP
        2: 130,   # Device
        3: 220,   # Sender Account
        4: 310,   # Receiver Account / Penampung
        5: 395    # Merchant
    }
    
    margin_x = 90
    
    for tier, nodes in tier_nodes.items():
        n_nodes = len(nodes)
        if n_nodes == 0:
            continue
        
        # Sort nodes by label or ID to keep layout consistent
        nodes.sort()
        
        # Calculate horizontal spacing
        if n_nodes == 1:
            coords[nodes[0]] = {
                "x": round(width / 2.0),
                "y": tier_y[tier]
            }
        else:
            spacing = (width - 2 * margin_x) / (n_nodes - 1)
            for idx, node_id in enumerate(nodes):
                coords[node_id] = {
                    "x": round(margin_x + idx * spacing),
                    "y": tier_y[tier]
                }
                
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
    
    # 4. Generate Coordinates via Hierarchical Flow Layout
    positions = generate_hierarchical_coordinates(G, nodes_info)
    
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
