import math

# coefficients
intercept = -0.50
coef_age = 0.02
coef_gender_male = 0.15
coef_chronic_conditions_yes = -0.25

# patient data
age = 40
gender_male = 1  # 1 for male, 0 for female
chronic_conditions_yes = 0  # 1 for yes, 0 for no

# calculate log-odds
log_odds = intercept + coef_age * age + coef_gender_male * gender_male + coef_chronic_conditions_yes * chronic_conditions_yes

# convert log-odds to probability
probability = math.exp(log_odds) / (1 + math.exp(log_odds))

print(probability)
