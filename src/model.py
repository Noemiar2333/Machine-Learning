import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

class SalesPredictor:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        self.prepare_data()
        self.model = LinearRegression()

    def prepare_data(self):
        self.data['Fecha pedido'] = pd.to_datetime(self.data['Fecha pedido'])
        self.data['Mes'] = self.data['Fecha pedido'].dt.month
        self.data['Año'] = self.data['Fecha pedido'].dt.year
        
        # Codificar la variable 'Tipo de producto'
        le = LabelEncoder()
        self.data['Tipo de producto'] = le.fit_transform(self.data['Tipo de producto'])

        self.X = self.data[['Mes', 'Año', 'Tipo de producto']]
        self.y = self.data['Unidades']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return y_pred, mse

    def predict_for_product(self, X_test_product):
        return self.model.predict(X_test_product)
