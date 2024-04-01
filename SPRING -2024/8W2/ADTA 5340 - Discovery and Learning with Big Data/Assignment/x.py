import pandas as pd

# Create the dataframe
data = {
    'Animal': ['African elephant', 'Thylacine', 'Two-toed sloth', 'Sea otter', 'Ring-tailed lemur', 'Aurochs', 'Common warthog', 'Brown bear', 'Orca'],
    'Mass_kg': [6000, 20, 6, None, 2.2, 700, 100, 200, 3000],
    'LifeSpan': [60, None, 20, 15, 17, None, 15, 25, 50],
    'BodyTemp': [35.9, None, 25, 37.8, 37, None, None, 37, 37],
    'Diet': ['Herbivore', 'Carnivore', 'Herbivore', 'Carnivore', 'Omnivore', 'Herbivore', 'Omnivore', 'Omnivore', 'Carnivore'],
    'ExtinctionDate': [None, 1936, None, None, None, 1627, None, None, None]
}

df = pd.DataFrame(data)

# Calculate the mean of LifeSpan
mean_life_span = df['LifeSpan'].mean()

print("Mean of LifeSpan:", mean_life_span)