# %%
# Import package
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load data
tips = sns.load_dataset('tips')
tips.columns = ['Total bill', 'Tip', 'Sex', 'Smoker', 'Day', 'Time', 'Party size']
tips

# %%

def create_custom_plot(data, x, y, plot_type, hue=None, style=None, size=None):
    """
    Creates a custom plot based on the specified type with support for hue, style, and size.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x: String, the column name to use for the x-axis.
    - y: String, the column name to use for the y-axis or the categorical variable for countplot.
    - plot_type: String, type of plot to create.
    - hue: (Optional) Variable in `data` to map plot aspects to different colors.
    - style: (Optional) Variable in `data` to map plot aspects to different markers.
    - size: (Optional) Variable in `data` to map plot aspects to different sizes.
    """
    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, hue=hue, style=style, size=size)
    elif plot_type == 'grouped_bar':
        sns.countplot(data=data, x=x, hue=y)  # Here `y` is used as hue for simplicity
    elif plot_type == 'strip':
        sns.stripplot(data=data, x=x, y=y, hue=hue)
    elif plot_type == 'swarm':
        sns.swarmplot(data=data, x=x, y=y, hue=hue)
    elif plot_type == 'kde':
        sns.kdeplot(data=data, x=x, hue=y)
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=x, y=y, hue=hue)
    elif plot_type == 'hist_grouped':
        sns.histplot(data=data, x=x, hue=y, multiple="dodge")
    elif plot_type == 'hist_stacked':
        sns.histplot(data=data, x=x, hue=y, multiple="stack")
    elif plot_type == 'faceted_scatter':
        sns.relplot(data=data, x=x, y=y, col=hue, hue=hue, size=size, kind="scatter")
    elif plot_type == 'faceted_count':
        sns.catplot(data=data, x=x, hue=y, col=style, kind="count", col_wrap=2)  # Assuming `style` is used for `col`
    else:
        print(f"Plot type '{plot_type}' is not supported.")
        return
    plt.show()

# Example usage
# Assuming 'tips' is a DataFrame already loaded with renamed columns as per your setup
# create_custom_plot(tips, 'Total bill', 'Tip', 'scatter', hue='Smoker', style='Smoker')

#%%
create_custom_plot(data=tips, x='Total bill', y='Tip',plot_type='scatter')
#%%
create_custom_plot(data=tips,  x='Day', y='Sex',plot_type='grouped_bar')
# %%
# Create scatter plot
sns.scatterplot(data=tips, x='Total bill', y='Tip')
# %%
# Create grouped bar plot with Day as the categorical variable
# grouped according to Sex
sns.countplot(data=tips, x='Day', hue='Sex')
# %%
# Create strip plot for total bill grouped according to day
sns.stripplot(data=tips, x='Total bill', y='Day')
# %%
# Create swarm plot for total bill grouped according to day
sns.swarmplot(data=tips, x='Day', y='Total bill')
# %%
# Create kde plot for total bill grouped according to day
sns.kdeplot(data=tips, x='Total bill', hue='Day')
# %%
# Create violin plot for total bill grouped according to day
sns.violinplot(data=tips, x="Day", y='Total bill')
# %%
# Create grouped bar chart for total bill grouped according to day
sns.histplot(data=tips, x='Total bill', hue='Day', shrink=0.8, multiple="dodge")
# %%
# Create stacked bar chart for total bill grouped according to day
sns.histplot(data=tips, x='Total bill', hue='Day', shrink=0.8, multiple="stack")
# %%
# Create scatter plot with color
sns.scatterplot(data=tips, x='Total bill', y='Tip', hue='Smoker', style='Smoker')
# %%
# Change opacity with alpha
sns.scatterplot(
    data=tips, x='Total bill', y='Tip', hue='Smoker', style='Smoker', alpha=0.7
)
# %%
# Change color palette with palette
sns.scatterplot(
    data=tips,
    x='Total bill',
    y='Tip',
    hue='Smoker',
    style='Smoker',
    alpha=0.7,
    palette='colorblind',
)

# %%
# Faceted scatter plot
sns.relplot(
    data=tips,
    x="Total bill",
    y="Tip",
    col="Sex",
    hue="Sex",
    size="Party size",
    style="Time",
    height=3,
)
# %%
# Faceted count plot
sns.catplot(
    data=tips,
    x="Time",
    hue="Smoker",
    col="Day",
    col_wrap=2,
    kind="count",
    height=2.5,
    aspect=0.6,
)
# %%
tips.hist()
# %%
tips.boxplot()
# %%
mpg = sns.load_dataset('mpg')
mpg.head()
# %%
# Create a new data frame with the columns "weight" and "mpg"
mpgSmall = mpg[['weight', 'mpg']]
# %%
print(mpgSmall)
# %%
# Create a scatter plot of weight vs mpg with x label "Weight" and y label "MPG"
sns.scatterplot(data=mpg, x='weight', y='mpg')

plt.savefig('mpg_scatter.png')
# %%
import plotly.express as px
px.scatter(mpg, x='weight' ,    y='mpg', size='cylinders')
# %%
