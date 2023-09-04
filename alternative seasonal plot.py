# Add a 'Season' column
data['Season'] = data.index.month.map(get_season)

# Extract years from the index
data['Year'] = data.index.year

years = data['Year'].unique()
seasons = ['Winter', 'Spring', 'Summer', 'Fall']

# seaborn
plt.figure(figsize=(10, 16))
sns.set_palette("colorblind")

for i, y in enumerate(years):
    if i > 0:
        seasonal_data = data[data['Year'] == y]
        sns.lineplot(x='Season', y='Rate', data=seasonal_data, label=str(y), marker='o', markersize=6)

# decoration
plt.ylim(min(data['Rate']) - 1, max(data['Rate']) + 1)
plt.ylabel('$Inflation$')
plt.xlabel('Season')
plt.title("Seasonal Plot of Inflation Time Series")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Year', fontsize=12, loc='best')
plt.show()
