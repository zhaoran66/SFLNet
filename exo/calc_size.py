import json
from collections import defaultdict

with open('egoexo4d_data/takes.json.63D2dC40', 'r') as f:
    data = json.load(f)

print(f'?? take ??: {len(data)}')
print()

scene_stats = defaultdict(lambda: {'count': 0, 'total_frames': 0})

for take in data:
    scenario = take.get('scenario', 'unknown')
    num_frames = take.get('nb_frames', 0) or 0
    scene_stats[scenario]['count'] += 1
    scene_stats[scenario]['total_frames'] += num_frames

print('?????????:')
print('-' * 60)
print('{:<15} {:>8} {:>15}'.format('????', 'take??', '?????'))
print('-' * 60)

for scene in sorted(scene_stats.keys()):
    stats = scene_stats[scene]
    print('{:<15} {:>8} {:>15,}'.format(scene, stats['count'], stats['total_frames']))

print()

avg_kb_per_frame = 15
print('?????? (256x256 RGB, JPEG):')
print('-' * 60)
for scene in sorted(scene_stats.keys()):
    stats = scene_stats[scene]
    total_gb = stats['total_frames'] * avg_kb_per_frame / 1024 / 1024
    print('{:<15} {:>10.1f} GB'.format(scene, total_gb))

print()
print('??????????? (cooking ????):')
print('-' * 60)
cooking_frames = scene_stats['cooking']['total_frames']
for res, kb in [('128x128', 5), ('256x256', 15), ('512x512', 50)]:
    total_gb = cooking_frames * kb / 1024 / 1024
    print('{}: {:.1f} GB'.format(res, total_gb))

print()
print('???: ?????????????/?????? (??????????)')
print('????????????: cooking ????? 300-500 GB')
