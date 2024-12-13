# Bajaj-Assessment-Answers
CODE for all the answers
import pandas as pd
from datetime import datetime
from collections import Counter

# Load the data
file_path = '/mnt/data/DataEngineeringQ2.json'
data = pd.read_json(file_path)

# Extract patientDetails and consultationData columns
patient_details = pd.json_normalize(data['patientDetails'])
consultation_data = pd.json_normalize(data['consultationData'])

# Question 1: Percentage of missing values
columns_to_check = ['firstName', 'lastName', 'birthDate']
total_rows = len(patient_details)
missing_percentages = {
    column: round((patient_details[column].isna().sum() + (patient_details[column] == '').sum()) / total_rows * 100, 2)
    for column in columns_to_check
}
print("Percentage of missing values:", missing_percentages)

# Question 2: Percentage of females after gender imputation
mode_gender = patient_details['gender'].mode()[0]
patient_details['gender'].fillna(mode_gender, inplace=True)
patient_details['gender'].replace('', mode_gender, inplace=True)
female_percentage = round((patient_details['gender'] == 'F').sum() / total_rows * 100, 2)
print("Percentage of females after imputation:", female_percentage)

# Question 3: Count of Adults
patient_details['age'] = patient_details['birthDate'].dropna().apply(
    lambda x: datetime.now().year - pd.to_datetime(x).year if pd.notnull(x) else None
)
def categorize_age(age):
    if pd.isnull(age):
        return None
    elif age <= 12:
        return "Child"
    elif 13 <= age <= 19:
        return "Teen"
    elif 20 <= age <= 59:
        return "Adult"
    else:
        return "Senior"
patient_details['ageGroup'] = patient_details['age'].apply(categorize_age)
adult_count = (patient_details['ageGroup'] == 'Adult').sum()
print("Count of Adults:", adult_count)

# Question 4: Average number of medicines prescribed
consultation_data['medicineCount'] = consultation_data['medicines'].apply(lambda meds: len(meds) if isinstance(meds, list) else 0)
average_medicines = consultation_data['medicineCount'].mean()
print("Average number of medicines prescribed:", average_medicines)

# Question 5: 3rd most frequently prescribed medicineName
all_medicines = consultation_data['medicines'].explode()
all_medicine_names = all_medicines.apply(lambda x: x['medicineName'] if isinstance(x, dict) and 'medicineName' in x else None).dropna()
medicine_counts = Counter(all_medicine_names)
third_most_frequent = medicine_counts.most_common(3)[-1][0]
print("3rd most frequently prescribed medicine:", third_most_frequent)

# Question 6: Percentage distribution of active and inactive medicines
active_count = all_medicines.apply(lambda x: x['isActive'] if isinstance(x, dict) and 'isActive' in x else None).sum()
inactive_count = len(all_medicines) - active_count
total_medicines = active_count + inactive_count
active_percentage = round((active_count / total_medicines) * 100, 2)
inactive_percentage = round((inactive_count / total_medicines) * 100, 2)
print("Active Medicines (%):", active_percentage, "Inactive Medicines (%):", inactive_percentage)

# Question 7: Valid phone numbers
def is_valid_indian_phone(number):
    number = str(number).strip()
    if number.startswith('+91'):
        number = number[3:]
    elif number.startswith('91'):
        number = number[2:]
    if len(number) == 10 and number.isdigit() and 6000000000 <= int(number) <= 9999999999:
        return True
    return False
patient_details['phoneNumber'] = data['phoneNumber']
patient_details['isValidMobile'] = patient_details['phoneNumber'].apply(is_valid_indian_phone)
valid_phone_count = patient_details['isValidMobile'].sum()
print("Count of valid phone numbers:", valid_phone_count)

# Question 8: Pearson correlation between medicines and age
merged_data = pd.concat([patient_details[['age']], consultation_data['medicineCount']], axis=1)
merged_data = merged_data.dropna()
correlation = merged_data['age'].corr(merged_data['medicineCount'])
print("Pearson correlation between age and number of medicines:", correlation)
