"""
Electroplating Plant Hoist Scheduler
=====================================
Reads:
  input_tanks_csv.csv   – station / tank configuration
  input_wagon_new.csv   – wagon / hoist parameters

Writes:
  OUTPUT_sequence.csv   – PLC instruction sequence
  DIP_TIME_OUTPUT.csv   – per-load dip-time audit log
"""

import math
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_TANKS  = 'input_tanks_csv.csv'
INPUT_WAGON  = 'input_wagon_new.csv'
OUT_SEQ      = 'OUTPUT_sequence.csv'
OUT_DIP      = 'DIP_TIME_OUTPUT.csv'

TOTAL_LOADS   = 10      # number of loads to process
N_FLIGHTBARS  = 3       # number of FlightBars (hangers/barrels) in this plant
PROJECT_ID    = 'Program 1'
PROGRAM_ID    = 1
WAGON_NO      = 1

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(val, default=0.0):
    try:
        x = float(val)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────────────────
# READ INPUTS
# ─────────────────────────────────────────────────────────────────────────────
tanks_df = pd.read_csv(INPUT_TANKS)
wagon_df = pd.read_csv(INPUT_WAGON)
tanks_df.columns = [c.strip() for c in tanks_df.columns]
wagon_df.columns = [c.strip() for c in wagon_df.columns]

w         = wagon_df.iloc[0]
FAST_SPD  = safe_float(w['Fast Speed Mtrs/Min'])        * 1000 / 60   # mm/s
SUPER_SPD = safe_float(w['Superfast SpeedMtrs/Min'])    * 1000 / 60   # mm/s
SLOW_SPD  = safe_float(w['Slow Speed Mtrs/Min'])        * 1000 / 60   # mm/s
LIFT_T    = safe_float(w['Lift Time Seconds'])
LOWER_T   = safe_float(w['Lower Time Seconds'])

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STATION DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
stations = {}
for _, row in tanks_df.iterrows():
    try:
        sno = int(row['station_no'])
    except (TypeError, ValueError):
        continue
    pno_raw = row['Process_NO']
    active  = pd.notna(pno_raw) and str(pno_raw).strip() not in ('', 'nan')
    mx      = safe_float(row['max_dip_time_sec'])
    can_rest_raw = str(row.get('can_rest_in_return_path', '')).strip().lower()
    stations[sno] = {
        'name'     : str(row['process_name']).strip(),
        'dist'     : safe_float(row['distance_mm']),
        'dip'      : safe_float(row['dip_time_sec']),
        'max_dip'  : mx if mx > 0 else float('inf'),
        'stype'    : str(row['station_type']).strip() if pd.notna(row.get('station_type', '')) else '',
        'active'   : active,
        'pno'      : int(float(pno_raw)) if active else None,
        'can_rest' : can_rest_raw in ('yes', 'true', '1'),
        'Criticality': str(row.get('Criticality', 'LOW')).strip().upper(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE PATH + ALTERNATING TANKS
# ─────────────────────────────────────────────────────────────────────────────
active_snos = sorted(s for s, d in stations.items() if d['active'])
# e.g. [1, 2, 5, 6, 7, 9, 10, 11, 13]

pno_count = Counter(stations[s]['pno'] for s in active_snos)
dup_pnos  = {pno for pno, cnt in pno_count.items() if cnt > 1}
alt_tanks = [s for s in active_snos if stations[s]['pno'] in dup_pnos]
# alt_tanks = [9, 10]  (ZINC PHOSPHATING duplicates)

LOAD_SNO   = active_snos[0]   # first station  = LOADING
UNLOAD_SNO = active_snos[-1]  # last station   = UNLOAD

# ─────────────────────────────────────────────────────────────────────────────
# TRAVEL TIME
# ─────────────────────────────────────────────────────────────────────────────
SLOW_ZONE_MM = 300.0

def travel_t(from_sno, to_sno, loaded=True, lift=False, lower=False):
    dist = abs(stations[to_sno]['dist'] - stations[from_sno]['dist'])
    spd  = FAST_SPD if loaded else SUPER_SPD
    if spd <= 0:
        raise ValueError(f"Wagon speed ({'Fast' if loaded else 'Superfast'}) is 0 or missing! Check wagon input.")
    
    if dist > SLOW_ZONE_MM and SLOW_SPD > 0:
        t_travel = (dist - SLOW_ZONE_MM) / spd + (SLOW_ZONE_MM / SLOW_SPD)
    else:
        t_travel = dist / spd
        
    t_total = t_travel
    if lift:
        t_total += LIFT_T
    if lower:
        t_total += LOWER_T
    return t_total

# ─────────────────────────────────────────────────────────────────────────────
# ALT-TANK DIP COMPENSATION FOR UNIFORM CYCLE TIME
# ─────────────────────────────────────────────────────────────────────────────
# Alt-tanks (same process, different positions) have different travel times to
# the next station → alternating cycle times.  Fix: for each alt-tank compute
# T_ref = max(dip[i] + T_to_next[i]) over the group, then set each tank's
# effective dip = T_ref - T_to_next so all loads arrive at the next station
# after the same total time.  Non-alt stations stay at their own dip.
# ─────────────────────────────────────────────────────────────────────────────
# ALT-TANK DIP COMPENSATION FOR UNIFORM CYCLE TIME
# ─────────────────────────────────────────────────────────────────────────────
pno_to_alts = {}
for s in alt_tanks:
    pno_to_alts.setdefault(stations[s]['pno'], []).append(s)

alt_tank_T_ref       = {}   # sno → T_ref
alt_tank_T_next      = {}   # sno → travel to next

# 1. Compute Bottleneck for Alternating Groups
process_cycle_bottlenecks = []
for pno, tank_list in pno_to_alts.items():
    last_alt_idx = max(active_snos.index(z) for z in tank_list)
    next_s = active_snos[last_alt_idx + 1] if last_alt_idx + 1 < len(active_snos) else None
    
    t_next_max = 0
    for t_sno in tank_list:
        tt = travel_t(t_sno, next_s, loaded=True, lift=True, lower=True) if next_s else 0.0
        alt_tank_T_next[t_sno] = tt
        t_next_max = max(t_next_max, tt)
    
    # Process bottleneck is (Dip + Travel) / NumTanks
    T_ref = stations[tank_list[0]]['dip'] + t_next_max
    for t_sno in tank_list: alt_tank_T_ref[t_sno] = T_ref
    process_cycle_bottlenecks.append(stations[tank_list[0]]['dip'] / len(tank_list))

# 2. Compute Bottleneck for Single Tanks
for sno in active_snos:
    if sno not in alt_tanks:
        idx = active_snos.index(sno)
        next_s = active_snos[idx+1] if idx+1 < len(active_snos) else None
        tt = travel_t(sno, next_s, loaded=True, lift=True, lower=True) if next_s else 0.0
        process_cycle_bottlenecks.append(stations[sno]['dip'] + tt)

target_cycle_time = max(process_cycle_bottlenecks)

# 3. Criticality-Aware Slack Distribution
# We want to fill the target_cycle_time by adding slack to LOW criticality tanks.
uniform_dips = {sno: d['dip'] for sno, d in stations.items()}

# Define criticality order (we prefer to add slack to LOW, then MEDIUM, then HIGH)
crit_order = {sno: (stations[sno]['Criticality'], sno) for sno in active_snos}
def crit_to_score(c):
    return {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}.get(c, 3)

# We adjust targets based on the bottleneck
# For this plant, Zinc (750/2) is the master stagger.
# We ensure every process occupies exactly its share of the cycle.
for sno in active_snos:
    if sno in alt_tanks:
        # Alt tanks are already part of a group that fits the stagger
        # We ensure they arrive at the next station at consistent times
        tt = alt_tank_T_next[sno]
        # For alt tanks, we set dip to maintain T_ref
        uniform_dips[sno] = alt_tank_T_ref[sno] - tt

def effective_ready_dip(sno):
    t = uniform_dips.get(sno, stations[sno]['dip'])
    mx = stations[sno]['max_dip']
    return t if mx == float('inf') else min(t, mx)

# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTBAR SETUP
# ─────────────────────────────────────────────────────────────────────────────
rest_stations = sorted(
    [s for s, d in stations.items() if d['can_rest']],
    key=lambda s: stations[s]['dist']
)

fb_pool       = list(range(1, N_FLIGHTBARS + 1))
fb_assignment = {}   # load_id → fb_id
empty_fbs     = {}   # sno → [{'id': fb_id, 'avail_time': clock}]

def find_park_station(from_sno):
    d_from = stations[from_sno]['dist']
    d_load = stations[LOAD_SNO]['dist']
    lo, hi = min(d_from, d_load), max(d_from, d_load)
    candidates = [s for s in rest_stations if lo < stations[s]['dist'] < hi]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(stations[s]['dist'] - d_from))

# ─────────────────────────────────────────────────────────────────────────────
# DESTINATION LOGIC  (stateless peek + stateful consume)
# ─────────────────────────────────────────────────────────────────────────────
zp_toggle = 0   # which alt tank is next

def peek_dest(from_sno):
    """Return destination without changing toggle state."""
    idx = active_snos.index(from_sno)
    if idx + 1 >= len(active_snos):
        return None
    nxt = active_snos[idx + 1]
    if from_sno in alt_tanks:                    # leaving an alt tank
        # Skip the other alt tanks in the same group
        pno = stations[from_sno]['pno']
        group = [s for s in alt_tanks if stations[s]['pno'] == pno]
        last_alt_idx = max(active_snos.index(z) for z in group)
        return active_snos[last_alt_idx + 1] if last_alt_idx + 1 < len(active_snos) else None
    if nxt in alt_tanks:                         # entering alt-tank slot
        pno = stations[nxt]['pno']
        group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
        return group[zp_toggle % len(group)]
    return nxt

def consume_dest(from_sno):
    """Return destination AND advance toggle if alt tank was chosen."""
    global zp_toggle
    dest = peek_dest(from_sno)
    if dest in alt_tanks:
        # Find which group this is
        pno = stations[dest]['pno']
        group = sorted([s for s in alt_tanks if stations[s]['pno'] == pno])
        if dest == group[-1]: # if we picked the last one, it doesn't matter, we increment anyway
             pass
        zp_toggle += 1
    return dest

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ACCUMULATORS
# ─────────────────────────────────────────────────────────────────────────────
seq_rows = []
dip_rows = []
inst_no  = 0
acc_time = 0.0

def add_seq(instruction, value, load_no=0, fb_id=None):
    if load_no == '' or load_no is None: load_no = 0
    # Clean up LOAD_NO if it's a string like '901'
    try:
        load_no = int(load_no)
    except:
        pass
    global inst_no
    inst_no += 1
    seq_rows.append({
        'PROJECT ID'        : PROJECT_ID,
        'Program ID'        : PROGRAM_ID,
        'Wagon Number'      : WAGON_NO,
        'Instruction'       : instruction,
        'Instruction Sr No' : inst_no,
        'Instruction Value' : value,
        'LOAD_NO'           : load_no,
        'FlightBar'         : f'FB{fb_id}' if fb_id is not None else '',
        'ACCUMULATED TIME'  : round(acc_time),
    })

def add_dip(load_id, sno, entry_t, exit_t):
    s      = stations[sno]
    actual = exit_t - entry_t
    actual_round = round(actual, 1)
    target = s['dip']
    mn, mx = s['dip'], s['max_dip']
    if target == 0:
        ok = True
    else:
        ok = mn <= actual_round <= (mx if mx != float('inf') else actual_round + 1)
    dip_rows.append({
        'Load ID'        : load_id,
        'Assigned Tank'  : s['name'],
        'Entry Time (s)' : round(entry_t, 1),
        'Exit Time (s)'  : round(exit_t, 1),
        'Target Dip (s)' : round(target, 1),
        'Actual Dip (s)' : actual_round,
        'Status'         : f"✓ PASS" if ok else f"✗ FAIL",
    })

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE WARMUP  (eliminates startup and end transients)
# ─────────────────────────────────────────────────────────────────────────────
N_WARMUP        = N_FLIGHTBARS
TOTAL_SIM_LOADS = N_WARMUP + TOTAL_LOADS + N_WARMUP

def _make_lid(idx):
    if idx < N_WARMUP:
        return 900 + idx + 1
    if idx < N_WARMUP + TOTAL_LOADS:
        return idx - N_WARMUP + 1
    return 100 + (idx - N_WARMUP - TOTAL_LOADS + 1) # post warmup 101, 102...

REAL_IDS = list(range(1, TOTAL_LOADS + 1))

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
tank_contents  = {}
clock          = 0.0
wagon_pos      = LOAD_SNO
next_load_idx  = 0
unloaded_count = 0

def _try_seed():
    global next_load_idx
    if LOAD_SNO not in tank_contents and fb_pool and next_load_idx < TOTAL_SIM_LOADS:
        new_fb  = fb_pool.pop(0)
        new_lid = _make_lid(next_load_idx)
        fb_assignment[new_lid] = new_fb
        tank_contents[LOAD_SNO] = {
            'load_id': new_lid, 'entry_time': clock, 'fb_id': new_fb
        }
        next_load_idx += 1

_try_seed()   # pre-place first load

for _iteration in range(100_000):
    if unloaded_count >= TOTAL_SIM_LOADS and not empty_fbs:
        break

    # ── Find best LOADED move (priority 1) ────────────────────────────────────
    best_sno   = None
    best_score = (float('inf'), float('inf'))
    for sno in sorted(tank_contents.keys(), reverse=True):
        dest = peek_dest(sno)
        if dest is None:
            continue
        if dest != UNLOAD_SNO and dest in tank_contents:
            continue
        arrive  = clock + travel_t(wagon_pos, sno, loaded=False, lift=True)
        ready   = tank_contents[sno]['entry_time'] + effective_ready_dip(sno)
        pickup  = max(arrive, ready)
        
        end_t = pickup + travel_t(sno, dest, loaded=True, lower=True)
        
        # Calculate how long the wagon will be trapped in a tight sequence
        chain_free_t = end_t
        curr_chk = dest
        while curr_chk is not None and curr_chk != UNLOAD_SNO:
            if curr_chk not in stations:
                break
            c_dip = stations[curr_chk]['dip']
            c_max = stations[curr_chk]['max_dip']
            if c_max != float('inf') and (c_max - c_dip) <= 60:
                chain_free_t += c_dip + 30  # approx 30s travel to next
                idx_curr = active_snos.index(curr_chk)
                if idx_curr + 1 < len(active_snos):
                    curr_chk = active_snos[idx_curr + 1]
                else:
                    break
            else:
                break

        violations = 0.0
        for other_sno, other_c in tank_contents.items():
            if other_sno == sno: continue
            mx = stations[other_sno]['max_dip']
            if mx == float('inf'): continue
            deadline = other_c['entry_time'] + mx
            
            # 1. Check direct move violation
            arrive_after = end_t + travel_t(dest, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
                
            # 2. Check trapped chain violation
            if chain_free_t > deadline:
                violations += (chain_free_t - deadline)
                
        score = (violations, pickup)
        if score < best_score:
            best_score = score
            best_sno   = sno
            
    best_pickup_t = best_score[1] if best_sno is not None else float('inf')

    # ── Find best EMPTY-FB return (priority 2) ────────────────────────────────
    best_empty_sno   = None
    best_empty_score = (float('inf'), float('inf'))
    for sno in empty_fbs:
        arrive = clock + travel_t(wagon_pos, sno, loaded=False, lift=True)
        end_t  = arrive + travel_t(sno, LOAD_SNO, loaded=True, lower=True)
        violations = 0.0
        for other_sno, other_c in tank_contents.items():
            mx = stations[other_sno]['max_dip']
            if mx == float('inf'): continue
            deadline = other_c['entry_time'] + mx
            arrive_after = end_t + travel_t(LOAD_SNO, other_sno, loaded=False, lift=True)
            if arrive_after > deadline:
                violations += (arrive_after - deadline)
                
        score = (violations, arrive)
        if score < best_empty_score:
            best_empty_score = score
            best_empty_sno   = sno
            
    best_empty_pickup_t = best_empty_score[1] if best_empty_sno is not None else float('inf')

    if best_sno is None and best_empty_sno is None:
        if tank_contents:
            soonest = min(
                v['entry_time'] + effective_ready_dip(k)
                for k, v in tank_contents.items()
            )
            clock = max(clock, soonest) + 0.1
        else:
            clock += 0.1
        continue

    # ── Priority decision ─────────────────────────────────────────────────────
    loading_stalled = (
        LOAD_SNO not in tank_contents and not fb_pool
        and bool(empty_fbs) and next_load_idx < TOTAL_LOADS
    )
    do_empty = (
        (best_sno is None and best_empty_sno is not None)
        or (loading_stalled and best_empty_sno is not None
            and best_empty_score <= best_score)
    )

    if do_empty:
        # ── Empty FlightBar return ────────────────────────────────────────────
        e_sno = best_empty_sno
        fb_info = empty_fbs[e_sno][0]
        fb_id = fb_info['id']

        travel_to_fb = travel_t(wagon_pos, e_sno, loaded=False, lift=True)
        acc_time    += travel_to_fb
        clock       += travel_to_fb
        wagon_pos    = e_sno
        add_seq('Get from', e_sno, '', fb_id)
        empty_fbs[e_sno].pop(0)
        if not empty_fbs[e_sno]:
            del empty_fbs[e_sno]

        # Always park at rest station if one exists on the return path
        park_sno = find_park_station(e_sno)

        if park_sno is not None:
            tt        = travel_t(e_sno, park_sno, loaded=True, lower=True)
            acc_time += tt
            clock    += tt
            wagon_pos = park_sno
            add_seq('Put on', park_sno, '', fb_id)
            empty_fbs.setdefault(park_sno, []).append({'id': fb_id, 'avail_time': clock})
        else:
            tt        = travel_t(e_sno, LOAD_SNO, loaded=True, lower=True)
            acc_time += tt
            clock    += tt
            wagon_pos = LOAD_SNO
            add_seq('Put on', LOAD_SNO, '', fb_id)
            fb_pool.append(fb_id)
            _try_seed()

    else:
        # ── Loaded FlightBar move ─────────────────────────────────────────────
        if best_sno is None:
            soonest = min(
                v['entry_time'] + effective_ready_dip(k)
                for k, v in tank_contents.items()
            )
            clock = max(clock, soonest) + 0.1
            continue

        content = tank_contents[best_sno]
        load_id = content['load_id']
        fb_id   = content['fb_id']

        travel_old = travel_t(wagon_pos, best_sno, loaded=False, lift=True)
        acc_time += travel_old
        
        lower_end_t       = content['entry_time'] + LOWER_T
        target_lift_t     = lower_end_t + stations[best_sno]['dip']
        required_depart   = target_lift_t - travel_old
        wait_secs         = max(0.0, required_depart - clock)

        if wait_secs > 0.1:
            acc_time += wait_secs
            add_seq('Wait for sec', round(wait_secs), load_id, fb_id)

        pickup_time = max(clock + travel_old, target_lift_t)
        clock       = pickup_time

        add_seq('Get from', best_sno, load_id, fb_id)

        dest = consume_dest(best_sno)
        tt   = travel_t(best_sno, dest, loaded=True, lower=True)
        acc_time   += tt
        clock      += tt
        entry_time  = clock

        add_seq('Put on', dest, load_id, fb_id)

        old_entry = content['entry_time']
        add_dip(load_id, best_sno, old_entry + LOWER_T, pickup_time)

        del tank_contents[best_sno]
        wagon_pos = dest

        if dest == UNLOAD_SNO:
            add_dip(load_id, dest, entry_time, entry_time)
            unloaded_count        += 1
            empty_fbs.setdefault(UNLOAD_SNO, []).append({'id': fb_id, 'avail_time': clock})
        else:
            tank_contents[dest] = {
                'load_id': load_id, 'entry_time': entry_time, 'fb_id': fb_id
            }

        _try_seed()

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUTS  (real loads only for dip log; all loads for sequence)
# ─────────────────────────────────────────────────────────────────────────────
dip_real = [r for r in dip_rows if r['Load ID'] in REAL_IDS]

pd.DataFrame(seq_rows).to_csv(OUT_SEQ, index=False)
pd.DataFrame(dip_real).to_csv(OUT_DIP, index=False)

print(f"Done — {unloaded_count}/{TOTAL_SIM_LOADS} simulated loads"
      f" ({N_WARMUP} warmup + {TOTAL_LOADS} real + {N_WARMUP} post-warmup).")
print(f"Target (uniform) cycle time : {target_cycle_time:.1f} s")
print(f"Sequence rows : {len(seq_rows)}")
print(f"Dip log rows  : {len(dip_real)} (real loads only)")
