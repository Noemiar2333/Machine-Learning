import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from model import SalesPredictor
from fpdf import FPDF

class SalesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Ventas")

        self.load_button = ttk.Button(root, text="Cargar archivo Excel", command=self.load_file)
        self.load_button.pack(pady=20)

        self.predict_button = ttk.Button(root, text="Predecir Ventas", command=self.predict_sales, state=tk.DISABLED)
        self.predict_button.pack(pady=20)

        self.generate_pdf_button = ttk.Button(root, text="Guardar Reporte de Ventas por Producto (PDF)", command=self.generate_pdf, state=tk.DISABLED)
        self.generate_pdf_button.pack(pady=20)

        self.result_label = ttk.Label(root, text="")
        self.result_label.pack(pady=20)

        self.report_text = tk.Text(root, height=15, width=50)
        self.report_text.pack(pady=20)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.predictor = SalesPredictor(file_path)
            self.predictor.train_model()
            self.predict_button.config(state=tk.NORMAL)
            self.generate_pdf_button.config(state=tk.NORMAL)
            messagebox.showinfo("Información", "Archivo cargado correctamente")

    def predict_sales(self):
        self.find_top_products()
        self.plot_predictions()
        self.generate_report()

    def find_top_products(self):
        df = self.predictor.data
        monthly_sales = df.groupby(['Mes', 'Tipo de producto'])['Unidades'].sum().reset_index()
        top_products = monthly_sales.groupby('Tipo de producto')['Unidades'].sum().nlargest(3).index.tolist()
        self.top_products = top_products

    def plot_predictions(self):
        df = self.predictor.data
        plt.figure(figsize=(15, 10))

        for i, product in enumerate(self.top_products, 1):
            product_data = df[df['Tipo de producto'] == product]
            monthly_sales = product_data.groupby('Mes')['Unidades'].sum().reindex(range(1, 13), fill_value=0)

            # Gráfico de ventas históricas
            plt.subplot(3, 2, 2*i-1)
            plt.bar(monthly_sales.index, monthly_sales.values, color='blue')
            plt.title(f'{product} - Ventas Históricas')
            plt.xlabel('Mes')
            plt.ylabel('Unidades Vendidas')
            plt.xticks(range(1, 13), [f'Mes {i}' for i in range(1, 13)])
            plt.grid(True, linestyle='--', alpha=0.7)

            # Predecir ventas para el producto
            product_x_test = self.predictor.X_test[self.predictor.X_test['Tipo de producto'] == product]
            y_pred = self.predictor.predict_for_product(product_x_test)

            # Compara las ventas predichas con las ventas del año pasado y asigna colores
            predicted_sales = pd.Series(y_pred).groupby(self.predictor.X_test[self.predictor.X_test['Tipo de producto'] == product].index // len(product_x_test)).sum()
            colors = ['green' if pred > actual else 'red' for pred, actual in zip(predicted_sales[:12], monthly_sales.values)]

            # Gráfico de ventas predichas
            plt.subplot(3, 2, 2*i)
            plt.bar(monthly_sales.index, predicted_sales[:12], color=colors)
            plt.title(f'{product} - Ventas Predichas')
            plt.xlabel('Mes')
            plt.ylabel('Unidades Vendidas')
            plt.xticks(range(1, 13), [f'Mes {i}' for i in range(1, 13)])
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        df = self.predictor.data
        monthly_sales = df.groupby(['Mes', 'Tipo de producto'])['Unidades'].sum().reset_index()

        # Generar reporte en texto
        report = "Reporte de Ventas por Producto\n\n"
        for month in range(1, 13):
            report += f"Mes {month}:\n"
            month_data = monthly_sales[monthly_sales['Mes'] == month]
            top_products = month_data.sort_values(by='Unidades', ascending=False)
            for _, row in top_products.iterrows():
                report += f"  Tipo de Producto: {row['Tipo de producto']}, Unidades Vendidas: {row['Unidades']}\n"
            report += "\n"

        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)

    def generate_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        df = self.predictor.data
        monthly_sales = df.groupby(['Mes', 'Tipo de producto'])['Unidades'].sum().reset_index()

        # Agregar título
        pdf.cell(200, 10, txt="Reporte de Ventas por Producto", ln=True, align='C')
        pdf.ln(10)

        # Agregar contenido
        for month in range(1, 13):
            pdf.cell(0, 10, txt=f"Mes {month}:", ln=True)
            month_data = monthly_sales[monthly_sales['Mes'] == month]
            top_products = month_data.sort_values(by='Unidades', ascending=False)
            for _, row in top_products.iterrows():
                pdf.cell(0, 10, txt=f"  Tipo de Producto: {row['Tipo de producto']}, Unidades Vendidas: {row['Unidades']}", ln=True)
            pdf.ln(10)

        # Guardar PDF
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            pdf.output(file_path)
            messagebox.showinfo("Información", f"PDF guardado en {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictionApp(root)
    root.mainloop()
