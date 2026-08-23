# -*- coding: utf-8 -*-
"""Analyze tensorboard logs from baseline training"""
import os
from tensorboard.backend.event_processing import event_accumulator

log_dir = '/data/data5/zhaoran/paper_code/baseline/logs'

# Find all event files
event_files = []
for f in os.listdir(log_dir):
    if 'events.out.tfevents' in f:
        event_files.append(os.path.join(log_dir, f))

print(f"Found {len(event_files)} event files:")
for f in event_files:
    print(f"  {os.path.basename(f)}")

# Load the most recent event file
if event_files:
    latest_event = sorted(event_files)[-1]
    print(f"\nLoading: {latest_event}")

    ea = event_accumulator.EventAccumulator(latest_event)
    ea.Reload()

    print(f"\nAvailable tags:")
    for tag in ea.Tags()['scalars']:
        print(f"  {tag}")

    # Get loss values
    tags_to_plot = ['Loss/train', 'Loss/val', 'Loss/interp_train', 'Loss/interp_val',
                    'MPJPE/train', 'MPJPE/val', 'PCK/train', 'PCK/val']

    print("\n" + "="*60)
    print("LOSS CURVE ANALYSIS")
    print("="*60)

    for tag in tags_to_plot:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                values = [(e.step, e.value) for e in events]
                print(f"\n{tag}:")
                print(f"  First:  step={values[0][0]}, value={values[0][1]:.4f}")
                print(f"  Last:   step={values[-1][0]}, value={values[-1][1]:.4f}")
                print(f"  Min:    step={min(values, key=lambda x:x[1])[0]}, value={min(v[1] for v in values):.4f}")
                print(f"  Max:    step={max(values, key=lambda x:x[1])[0]}, value={max(v[1] for v in values):.4f}")
