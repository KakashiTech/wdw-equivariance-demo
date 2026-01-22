#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Example demonstration values to visualize the micro-test outcome
labels = ['Baseline', 'After break', 'After repair']
values = [0.0, 4.444e-02, 0.0]

fig, ax = plt.subplots(figsize=(4,3))
bars = ax.bar(labels, values, color=['#4caf50', '#f44336', '#2196f3'])
ax.set_ylabel('Equivariance error')
ax.set_title('Micro-test: equivariance (n=3)')
ax.set_ylim(0, max(values)*1.5 if max(values)>0 else 0.05)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + (max(values)*0.03 if max(values)>0 else 0.01), f'{v:.2e}', ha='center')
plt.tight_layout()
plt.savefig('docs/figs/equivariance_bar.png', dpi=150)
print('Saved docs/figs/equivariance_bar.png')
