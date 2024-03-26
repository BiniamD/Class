# Load necessary package
library(stats)
library(readxl)

file <- "c:/Users/17034/downloads/Spring 2024 Airline Dataset.xlsx"
# Read the data from the Excel file
airline_data <- read_excel(file, sheet = "Dataset")

# View the data
View(airline_data)

# Check the structure of the data
str(airline_data)

# Check the summary of the data
summary(airline_data)

# Check the number of rows and columns in the data
nrow(airline_data)
ncol(airline_data)

# Check the column names of the data
colnames(airline_data)

# Check the type of data in each column
sapply(airline_data, class)

# Check the number of unique values in each column
sapply(airline_data, function(x) length(unique(x)))

# Check the number of missing values in each column
sapply(airline_data, function(x) sum(is.na(x)))

unique(airline_data$DayOfWeek)



# Convert all character columns to factors
airline_data[ , sapply(airline_data
    # Select all character columns
    , is.character)] <- lapply(airline_data[ 
    # Select all character columns
    , sapply(airline_data, is.character)]
    # Convert to factors
    , as.factor)

# Create dummy variables for all factor columns
dummy_vars <- model.matrix(~ . - 1, airline_data)

# Create a new data frame with the dummy variables
dummy_vars <- as.data.frame(dummy_vars)

# View the new data frame
View(dummy_vars)

#columns
colnames(dummy_vars)





# Load necessary package
library(stats)

# Assuming 'flight_data' is your dataset with 'price' and 'airline' columns
# 'price' should be numeric, and 'airline' should be a factor or categorical variable

# First, ensure 'airline' is treated as a factor
airline_data$Airline <- as.factor(airline_data$Airline)

# Conduct ANOVA to compare mean flight prices across different airlines
result <- aov(FlightPrice ~ Airline, data = airline_data)
summary(result)

#Interpreting the Result:
#P-value: The key metric to look for in the ANOVA summary output. If the p-value associated with the 'airline' variable is less than your chosen significance level (typically 0.05), you reject the null hypothesis. This indicates that there are significant differences in the mean prices among the airlines.
#F-statistic: Provides a measure of the overall variance explained by the categorical variable 'airline' compared to the unexplained variance. A higher value suggests a stronger effect.
#If you find significant differences (p-value < 0.05), it suggests that not all airlines charge the same average prices, warranting further investigation into which specific airlines differ. This can be followed up with post-hoc tests (e.g., Tukey's HSD) to identify specific groups (airlines) between which these price differences occur.
#This analysis provides insights into pricing strategies, potentially reflecting differences in service quality, target markets, or operational costs across airlines.


# ANOVA Test with interaction between FlightType (International vs Domestic) and DayOfWeek
aov_result2 <- aov(FlightPrice ~ FlightType * DayOfWeek, data = airline_data)
summary(aov_result2)



################################################################

# Multiple Linear Regression with interaction term
lm_result <- lm(FlightPrice ~ FlightDuration, data = airline_data)
summary(lm_result)



# Multiple Linear Regression with interaction term
lm_result2<- lm(FlightPrice ~ FlightDuration, data = airline_data)
summary(lm_result2)


destination_dummies <- model.matrix(~ DestinationAirport - 1, data = airline_data)
# Construct the formula string
# Include 'IsHolidaySeason' and dynamically add all destination dummy variables
formula_str <- paste("FlightPrice ~ HolidayperioudYes +", paste(names(destination_dummies), collapse=" + "))

# Convert the string to a formula object
formula_obj <- as.formula(formula_str)

# Fit the model
model <- lm(formula_obj, data = flight_data)

# Summary of the model
summary(model)

# ANOVA Test for the effect of HolidaySeason and Destination on FlightPrice
lm_result3 <- lm(FlightPrice ~ HolidayperioudYes + DestinationAirport* , data = dummy_vars)
summary(lm_result3)

# Multiple Regression Testing considering various predictors
#lm_result2 <- lm(FlightPrice ~ FlightDuration + Distance + TimeOfDay + DayOfWeek, data = airline_data)
#summary(lm_result2)

################################################################



# Run the regression
model <- lm(FlightPrice ~ ., data = dummy_vars)  # X1 is the name of the outcome column

# Summary of the model
summary(model)


# Identify significant predictors
significant_predictors <- summary(model)$coefficients[which(summary(model)$coefficients[, 4] < 0.05), ]
print(significant_predictors)  # Print the significant predictors


model2 <- lm(formula = FlightPrice ~ DayOfWeekMonday + DayOfWeekTuesday + DayOfWeekWednesday + DayOfWeekThursday  + DayOfWeekSaturday + DayOfWeekSunday, data = dummy_vars)
summary(model2)


# Identify significant predictors
significant_predictors <- summary(model2)$coefficients[which(summary(model2)$coefficients[, 4] < 0.05), ]
print(significant_predictors)  # Print the significant predictors


model2 <- lm(formula = FlightPrice ~ FuelSurcharge, data = airline_data)
summary(model2)

plot(model2)





#result <- fastDummies::dummy_cols(airline_data, remove_first_dummy =TRUE,remove_selected_columns = TRUE)View(result)



# Assuming 'flight_data' is your dataset with 'price' and 'airline' columns
# 'price' should be numeric, and 'airline' should be a factor or categorical variable

# First, ensure 'airline' is treated as a factor
airline_data$Airline <- as.factor(airline_data$Airline)

# Conduct ANOVA to compare mean flight prices across different airlines
result <- aov(FlightPrice ~ Airline, data = airline_data)
summary(result)




# ANOVA Test
# Load necessary libraries
library(ggplot2)


aov_result <- aov(FlightPrice ~ Airline, data=airline_data)
summary(aov_result)

options(scipen=999)
# Regression Analysis
lm_result <- lm(FlightPrice ~ FlightDuration + Airline, data=airline_data)
summary(lm_result)
x
# ANOVA Test with interaction between flight type and day of the week

aov_result2 <- aov(FlightPrice ~ ClassType * DayOfWeek, data=airline_data)
summary(aov_result2)


# Multiple Regression Testing considering various predictors
lm_result2 <- lm(FlightPrice ~ FlightDuration + Distance + TimeOfDay + DayOfWeek, data = airline_data)
summary(lm_result2)

install.packages("fastDummies");
library(fastDummies);

airline_data <- dummy_cols(airline_data, select_columns = 'DayOfWeek')
airline_data <- dummy_cols(airline_data, select_columns = 'TimeOfDay')
airline_data <- dummy_cols(airline_data, select_columns = 'ClassType')
airline_data <- dummy_cols(airline_data, select_columns = 'Airline')


#Kaplan, J. & Schlegel, B. (2023). fastDummies: Fast Creation of Dummy (Binary) Columns and Rows from Categorical Variables. Version 1.7.1. URL: https://github.com/jacobkap/fastDummies, https://jacobkap.github.io/fastDummies/.
