while True:
 try:
  i=input("
[You] ");
  if i=="quit":break
  elif i=="status":
   s=core.get_status();sub=s["substrate"];
   print(f"📊 Status: {sub["conversation_count"]} convs, {sub["node_count"]} nodes, {sub["edge_count"]} edges, {sub["community_count"]} communities, Φ={s["metrics"]["phi_value"]:.3f}");continue
  st=time.time();r=core.process_input(i);pt=time.time()-st;
  print(f"
[ISC-AI] {r}");s=core.get_status();
  print(f"
[Φ={s["metrics"]["phi_value"]:.3f}, Nodes={s["substrate"]["node_count"]}, t={pt:.3f}s]")
 except KeyboardInterrupt:
  print("
Goodbye!");break
 except Exception as e:
  print(f"Error: {e}")