# Install required packages (if needed)
!pip install faker pandas numpy

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()
random.seed(42)
np.random.seed(42)

# Kenyan configurations
KENYAN_NAMES = [    
    'Wanjiku', 'Mwende', 'Kamau', 'Njoroge', 'Omondi', 'Atieno', 'Kipchoge', 'Chebet', 'Maina', 'Nyambura',
    'Mutai', 'Kibet', 'Jepchirchir', 'Adhiambo', 'Wambui', 'Karanja', 'Barasa', 'Wainaina', 'Wakaba', 'Omolo',
    'Ochieng', 'Otieno', 'Akinyi', 'Achieng', 'Atwoli', 'Amina', 'Fatuma', 'Salma', 'Mohamed', 'Abdi',
    'Ibrahim', 'Ali', 'Khadija', 'Njeri', 'Muturi', 'Mumbi', 'Chepchumba', 'Wambui', 'Makena', 'Mwende',
    'Onyango', 'Odhiambo', 'Kariuki', 'Muthoni', 'Wairimu', 'Ndungu', 'Korir', 'Juma', 'Akinyi', 'Wambui',
    'Mwangi', 'Ochieng', 'Kamau', 'Mutua', 'Kiplagat', 'Kibet', 'Kiprotich', 'Cheruiyot', 'Koech', 'Langat',
    'Simiyu', 'Chepkoech', 'Mwai', 'Kimenyi', 'Mugambi', 'Muli', 'Mugo', 'Waweru', 'Omondi', 'Muriithi',
    'Nyambura', 'Makena', 'Anyango', 'Waithera', 'Odera', 'Mutheu', 'Mbugua', 'Kilonzo', 'Karanja', 'Chebet'
]

KENYAN_BANKS = [
    'Equity Bank',
    'KCB Bank',
    'Co-operative Bank',
    'Absa Kenya',
    'Stanbic Bank','NCBA Bank','Diamond Trust Bank',
    'Standard Chartered Kenya',
    'I&M Bank',
    'Family Bank',
    'Bank of Africa Kenya',
    'Citibank N.A Kenya',
    'Ecobank Kenya',
    'National Bank of Kenya',
    'Prime Bank',
    'Gulf African Bank',
    'Victoria Commercial Bank',
    'Housing Finance Company of Kenya',
    'Commercial Bank of Africa',
    'Chase Bank Kenya',
    'Imperial Bank Kenya',
    'Jamii Bora Bank',
    'NIC Bank',
    'Sidian Bank',
    'Spire Bank',
    'Transnational Bank Kenya'
]

CARD_TYPES = [
    'Visa Classic',
    'MasterCard Gold',
    'Visa Platinum',
    'Equity card',
    'Visa Gold',
    'MasterCard Standard',
    'Visa Infinite',
    'MasterCard Platinum',
    'Visa Signature',
    'MasterCard World Elite'
]

KENYAN_LOCATIONS = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret',
    'Thika', 'Meru', 'Nyeri', 'Kakamega', 'Machakos',
    'Ruiru', 'Kikuyu', 'Malindi', 'Kitale', 'Garissa',
    'Embu', 'Kericho', 'Kilifi', 'Homa Bay', 'Naivasha',
    'Lamu', 'Migori', 'Bungoma', 'Narok', 'Nanyuki'
]

FRAUD_LOCATIONS = [
    'Kampala, Uganda', 'Kigali, Rwanda', 'Dar es Salaam', 
    'Juba, South Sudan', 'London, UK', 'Dubai, UAE', 'Ireland',
    'United Kingdom', 'France', 'Luxembourg', 'Malta', 'Denmark',
    'Sweden', 'Spain', 'Netherlands', 'USA'
]

MERCHANT_CATEGORIES = {
    'Retail': ['Nakumatt Supermarket', 'Tuskys', 'Naivas', 'Quickmart'],
    'Travel': ['Jambojet', 'Kenya Airways', 'Mombasa Safari Tours'],
    'Electronics': ['Safaricom Shop', 'Phone World', 'Tech Africa'],
    'Suspicious': ['Online Forex', 'Crypto Exchange', 'Gambling Site']
}

def generate_transactions(num_records):
    users = {}
    data = []
    start_date = datetime(2023, 1, 1)
    
    # Non-linear probability curve parameters
    base_growth = 0.04  # 4% starting point
    max_growth = 0.08   # 8% maximum
    curve_sharpness = 0.0001  # Controls how fast probability increases
    
    for idx in range(num_records):
        # Base transaction
        trans_date = start_date + timedelta(minutes=random.randint(1, 60*24))
        trans_id = f"TXN{idx+1:07d}"
        name = random.choice(KENYAN_NAMES)
        
        # Initialize user if new
        if name not in users:
            users[name] = {
                'bank': random.choice(KENYAN_BANKS),
                'card': random.choice(CARD_TYPES),
                'usual_location': random.choice(KENYAN_LOCATIONS),
                'txn_count': 0,
                'last_txn': trans_date - timedelta(days=7),
                'fraud_risk': random.betavariate(1.5, 3)  # User-specific risk factor
            }
        
        # Complex probability calculation (FIXED)
        progress = idx/num_records  # 0-1 scale
        user_risk = users[name]['fraud_risk']
        
        # Sigmoid-based growth with random noise (ADDED MISSING PARENTHESIS)
        base_prob = base_growth + (max_growth - base_growth) / (
            1 + np.exp(-curve_sharpness * (idx - num_records/2)))
        
        # Add multiple noise components
        time_noise = 0.02 * np.sin(2 * np.pi * progress * 10)
        user_noise = 0.03 * (user_risk - 0.3)
        random_noise = 0.02 * random.uniform(-1, 1)
        
        current_prob = np.clip(
            base_prob + time_noise + user_noise + random_noise,
            0.03, 0.09
        )
        
        is_fraud = 0
        if random.random() < current_prob:
            is_fraud = 1
            
            # Masked fraud patterns
            location = (users[name]['usual_location'] 
                      if random.random() < 0.4 + 0.3*progress
                      else random.choice(FRAUD_LOCATIONS))
            
            amount = round(min(
                abs(np.random.normal(50000 + 100000*progress, 75000)), 
                500000
            ), 2)
            
            # Dynamic merchant selection
            merchant = (random.choice(MERCHANT_CATEGORIES['Suspicious'])
                      if progress < 0.8 else 
                      random.choice(MERCHANT_CATEGORIES[random.choice(['Retail','Travel'])]))
            
            # Bank switching logic
            bank = (users[name]['bank'] 
                   if random.random() < 0.7 - 0.3*progress
                   else random.choice([b for b in KENYAN_BANKS if b != users[name]['bank']]))
            
            # Transaction type definition
            txn_type = 'ATM' if random.random() > 0.4 else 'Online'
            
            # Time patterns
            time_since_last = round(random.uniform(
                0.02 + 0.1*progress,
                0.5 - 0.3*progress
            ), 2)
            
            txn_freq = random.randint(
                max(5, int(15 - 10*progress)),
                min(25, int(20 + 10*progress))
            )
            
        else:
            # Normal transaction patterns
            location = (users[name]['usual_location']
                      if random.random() < 0.9 - 0.2*progress
                      else random.choice(KENYAN_LOCATIONS))
            
            amount = round(min(
                abs(np.random.normal(8000 + 2000*progress, 3000)),
                100000
            ), 2)
            
            merchant = random.choice(
                MERCHANT_CATEGORIES[random.choice(['Retail','Travel','Electronics'])]
            )
            
            # Transaction type definition
            txn_type = random.choice(['Retail', 'Online', 'ATM'])
            
            bank = users[name]['bank']
            time_since_last = round(random.uniform(
                1 + 2*progress,
                24 - 6*progress
            ), 2)
            
            txn_freq = random.randint(
                max(1, int(3 - 1*progress)),
                min(7, int(5 + 2*progress))
            )
        
        # Add delayed fraud effects
        if idx > 10000 and idx % 500 == 0:  # Periodic pattern
            if random.random() < 0.05:
                is_fraud = 1 - is_fraud  # Flip status for some transactions
                
        # Update user profile
        users[name]['txn_count'] += 1
        users[name]['last_txn'] = trans_date
        
        # Build record
        record = {
            'transaction_id': trans_id,
            'user_name': name,
            'credit_card_type': users[name]['card'],
            'transaction_amount': amount,
            'merchant_category': merchant,
            'datetime': trans_date.strftime("%Y-%m-%d %H:%M"),
            'bank': bank,
            'location': location,
            'is_foreign': int(location not in KENYAN_LOCATIONS),
            'transaction_type': txn_type,
            'transaction_frequency': txn_freq,
            'time_since_last_txn': time_since_last,
            'user_tenure': users[name]['txn_count'],  # Important masked feature
            'hour_of_day': trans_date.hour,
            'is_fraud': is_fraud
        }
        
        data.append(record)
        
        if (idx+1) % 10000 == 0:
            print(f"Generated {idx+1} transactions...")
    
    return pd.DataFrame(data)

def add_realism(df):
    """Add sophisticated noise while preserving underlying patterns"""
    # 1. Temporal noise with periodic patterns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
        
    df['time_since_last_txn'] = df['time_since_last_txn'].apply(
    lambda x: max(1, min(x * random.uniform(0.9, 1.1), 1440))  # Added closing )
)
    
    # 2. Dynamic amount noise based on transaction context
    df['transaction_amount'] = df.apply(
        lambda x: round(
            x['transaction_amount'] * 
            np.random.normal(1.02, 0.03 if x['is_fraud'] else 0.01),
            2
        ), axis=1
    ).clip(lower=10, upper=500000)
    
    # 3. Location obfuscation with time-dependent patterns
    for idx, row in df.iterrows():
        if row['is_fraud']:
            if random.random() < 0.15 * (1 - idx/len(df)):  # Decreasing fake locals
                df.at[idx, 'location'] = random.choice(KENYAN_LOCATIONS)
        else:
            if random.random() < 0.005 + 0.015*(idx/len(df)):  # Increasing fake foreigners
                df.at[idx, 'location'] = random.choice(FRAUD_LOCATIONS)
    
    # 4. Non-linear time perturbation
    df['time_since_last_txn'] = df.apply(
        lambda x: max(0.02, min(
            x['time_since_last_txn'] * 
            np.random.lognormal(0, 0.1 if x['is_fraud'] else 0.05),
            72
        )), axis=1
    ).round(2)
    
    # 5. Dynamic label noise with confusion spikes
    df['original_fraud'] = df['is_fraud'].copy()
    for hour in range(24):
        hour_mask = df['hour'] == hour
        flip_prob = 0.03 + 0.02*np.sin(hour * np.pi/6)  # Time-dependent noise
        df.loc[hour_mask, 'is_fraud'] = df.loc[hour_mask, 'is_fraud'].apply(
            lambda x: x if random.random() < 0.95 else int(random.random() < flip_prob)
        )
    
    # 6. Merchant category evolution
    valid_merchants = sum(MERCHANT_CATEGORIES.values(), [])
    df['merchant_category'] = df.apply(
        lambda x: (x['merchant_category'] if random.random() < 0.9 
                  else random.choice(valid_merchants[-3:] if x['is_fraud'] 
                                   else valid_merchants[:-3])),
        axis=1
    )
    
    # 7. Bank switching with memory
    last_banks = {}
    df['bank'] = df.apply(
        lambda x: (x['bank'] 
                  if random.random() < 0.85 - 0.1*(x['is_fraud']) 
                  else (last_banks.get(x['user_name'], x['bank']) 
                       if random.random() < 0.3 
                       else random.choice(KENYAN_BANKS))),
        axis=1
    )
    # Update bank memory
    for idx, row in df.iterrows():
        last_banks[row['user_name']] = row['bank']
    
    # 8. Frequency patterns with drift
    df['transaction_frequency'] = df.apply(
        lambda x: max(1, min(
            x['transaction_frequency'] + 
            np.random.randint(-1, 2) + 
            int(0.5 * (idx/len(df))),  # Gradual drift
            30
        )), 
        axis=1
    )
    
    # 9. Context-aware foreign flag
    df['is_foreign'] = df.apply(
        lambda x: int((x['location'] not in KENYAN_LOCATIONS) ^ 
                     (random.random() < 0.05 - 0.03*(x['is_fraud']))),
        axis=1
    )
    
    # 10. Cleanup and validation
    df = df.drop(columns=['day_of_week', 'original_fraud'])
    df = df.dropna(subset=['transaction_amount', 'user_name'])
    df = df[df['transaction_amount'] > 0]
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Generate 50,000 transactions
df = generate_transactions(500000)
df = add_realism(df)

# Add some noise to make patterns less obvious
df['transaction_amount'] = df['transaction_amount'].apply(
    lambda x: x * random.uniform(0.9, 1.1))
df['transaction_frequency'] = df['transaction_frequency'].apply(
    lambda x: max(1, x + random.randint(-2, 2)))

# Save to CSV
df.to_csv('realistic_kenyan_fraud_data.csv', index=False)
print(f"Dataset saved! Fraud rate: {df.is_fraud.mean():.2%}")


# Generate base data
df = generate_transactions(500000)

# Add sophisticated realism
df = add_realism(df)

# Final validation
print(f"Final fraud rate: {df.is_fraud.mean():.2%}")
print(f"Fraud/feature correlations:")
print(df.corr()['is_fraud'].sort_values(ascending=False))

# Save to CSV
df.to_csv('kenyan_fraud_obfuscated.csv', index=False)
