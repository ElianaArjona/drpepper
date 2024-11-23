from openai import OpenAI
import yaml
import pandas as pd
import re

class ModelConfig:
    def __init__(self, config_path) -> None:
        self.model_name = None
        self.stream = None
        self.temperature = None
        self.role = None
        self.content = None
        self.path = config_path
    
    def get_configuration(self):
        with open(self.path, 'r') as file:
            config = yaml.safe_load(file)

        self.model_name = config['stream']['model']
        self.stream = config['stream']['stream']
        self.temperature = config['stream']['temperature']
        self.role = config['stream']['messages'][0]['role']

class OpenAIModel:
    def __init__(self, config) -> None:
        self.client = OpenAI()
        self.config = config
    
    def get_inference(self, content):
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": self.config.role,
                "content": content
                }],
            stream=self.config.stream,
            temperature=self.config.temperature
        )
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
        
        return response_content

def run():
    config = ModelConfig(
            config_path='./model/config/model_config.yaml')
    config.get_configuration()

    with open('./model/data/sample.txt', 'r') as file:
        user_input = file.read().strip()

    action = "Clasificas información en una tabla con las siguientes columnas: \
        Fecha, Broker, Ubicación, Recamaras, Baños,\
        Estacionamiento, Balcón, Precio, metros cuadrados, Detalles extra\
        Numero de Telefono, Alquiler/Venta, Amoblado/Linea Blanca, \
        Operación Requerida (Busca, Ofrece), \
        Tipo de Propiedad (casa, apartamento, local, oficina, terreno)"
    
    input = user_input.join([action, "\n", user_input])

    # print(input)

    llm = OpenAIModel(config=config)
    markdown = llm.get_inference(input)

    table_content = re.findall(r'\|(.+)\|', markdown)

    # Convert Markdown table to Pandas DataFrame
    rows = []
    for line in table_content:
        row = [cell.strip() for cell in line.split('|') if cell.strip()]
        rows.append(row)

    df = pd.DataFrame(rows[1:], columns=rows[0])

    print(df)

        
run()

