import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load your dataset
# df = pd.read_csv('orders.csv')

# 2. Select features (X) and target (y)
# Note: We EXCLUDE 'customer_id' and 'discount_used_on_repeat_order'
X = df[['order_count_last_90d', 'avg_order_value', 'days_since_last_order']]
y = df['repeat_purchase_flag']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Generate the Output (Predictions)
predictions = model.predict(X_test)

# 6. Evaluate
print(classification_report(y_test, predictions))
