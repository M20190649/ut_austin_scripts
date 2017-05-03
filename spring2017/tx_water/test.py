import pandas as pd
import matplotlib.pyplot as plt

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'pre_score': [4, 24, 31, 2, 3],
        'mid_score': [25, 94, 57, 62, 70],
        'post_score': [5, 43, 23, 23, 51]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])

print df

# Create a figure with a single subplot
f, ax = plt.subplots(1, figsize=(10,5))

# Set bar width at 1
bar_width = 1

# positions of the left bar-boundaries
bar_l = [i for i in range(len(df['pre_score']))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]

# Create the total score for each participant
totals = [i+j+k for i,j,k in zip(df['pre_score'], df['mid_score'], df['post_score'])]
print totals

# Create the percentage of the total score the pre_score value for each participant was
pre_rel = [i / float(j) * 100 for  i,j in zip(df['pre_score'], totals)]
print pre_rel

# Create the percentage of the total score the mid_score value for each participant was
mid_rel = [i / float(j) * 100 for  i,j in zip(df['mid_score'], totals)]
print mid_rel

# Create the percentage of the total score the post_score value for each participant was
post_rel = [i / float(j) * 100 for  i,j in zip(df['post_score'], totals)]
print post_rel

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using pre_rel data
       pre_rel,
       # labeled
       label='Pre Score',
       # with alpha
       alpha=0.9,
       # with color
       color='#019600',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using mid_rel data
       mid_rel,
       # with pre_rel
       bottom=pre_rel,
       # labeled
       label='Mid Score',
       # with alpha
       alpha=0.9,
       # with color
       color='#3C5F5A',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using post_rel data
       post_rel,
       # with pre_rel and mid_rel on bottom
       bottom=[i+j for i,j in zip(pre_rel, mid_rel)],
       # labeled
       label='Post Score',
       # with alpha
       alpha=0.9,
       # with color
       color='#219AD8',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Set the ticks to be first names
plt.xticks(tick_pos, df['first_name'])
ax.set_ylabel("Percentage")
ax.set_xlabel("")

# Let the borders of the graphic
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(-10, 110)

# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

# shot plot
plt.show()