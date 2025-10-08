#!/usr/bin/env python3
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from datetime import datetime
import numpy as np

class IrrigationDatabase:
    def __init__(self, host='localhost', database='irrigation_db', 
                 user='postgres', password='postgres', port=5432):
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Estabelece conexão com o banco de dados"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            print(f"[OK] Conectado ao banco: {self.connection_params['database']}")
            return True
        except Exception as e:
            print(f"[ERRO] Falha na conexão: {e}")
            return False

    def disconnect(self):
        """Fecha a conexão com o banco"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("[OK] Conexão fechada")

    def create_tables(self):
        """Cria as tabelas do banco de dados"""
        
        # Gerar colunas de lâmina para cada ângulo (0-359)
        lamina_columns = []
        for angle in range(360):
            lamina_columns.append(f"lamina_at_{angle:03d} DECIMAL(10,4) DEFAULT 0.0")
        
        lamina_columns_sql = ",\n            ".join(lamina_columns)
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS cycle_lamina_data (
            id SERIAL PRIMARY KEY,
            cycle_id VARCHAR(100) NOT NULL,
            pivo_id VARCHAR(50) NOT NULL,
            start_date TIMESTAMP NOT NULL,
            end_date TIMESTAMP NOT NULL,
            blade_factor DECIMAL(8,4) DEFAULT 5.46,
            duration_minutes INTEGER,
            total_angles_covered INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Valores de lâmina para cada ângulo (0-359 graus)
            {lamina_columns_sql},
            
            -- Restrições
            CONSTRAINT unique_cycle_pivo UNIQUE(cycle_id, pivo_id),
            CONSTRAINT valid_duration CHECK(duration_minutes >= 5),
            CONSTRAINT valid_blade_factor CHECK(blade_factor > 0)
        );
        """
        
        try:
            self.cursor.execute(create_table_sql)
            self.conn.commit()
            print("[OK] Tabela 'cycle_lamina_data' criada com sucesso")
            
            # Criar índices
            self._create_indexes()
            
        except Exception as e:
            print(f"[ERRO] Falha ao criar tabela: {e}")
            self.conn.rollback()

    def _create_indexes(self):
        """Cria índices para otimização de consultas"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_pivo_date ON cycle_lamina_data(pivo_id, start_date DESC);",
            "CREATE INDEX IF NOT EXISTS idx_start_date ON cycle_lamina_data(start_date DESC);",
            "CREATE INDEX IF NOT EXISTS idx_pivo_id ON cycle_lamina_data(pivo_id);",
            "CREATE INDEX IF NOT EXISTS idx_cycle_id ON cycle_lamina_data(cycle_id);"
        ]
        
        for index_sql in indexes:
            try:
                self.cursor.execute(index_sql)
            except Exception as e:
                print(f"[AVISO] Índice já existe ou erro: {e}")
        
        self.conn.commit()
        print("[OK] Índices criados")

    def insert_cycle_data(self, cycle_data):
        """
        Insere dados de um ciclo individual
        
        cycle_data deve ser um dict com:
        {
            'cycle_id': str,
            'pivo_id': str,
            'start_date': datetime,
            'end_date': datetime,
            'blade_factor': float,
            'duration_minutes': int,
            'lamina_360': np.array[360]  # valores de lâmina para cada ângulo
        }
        """
        
        # Preparar colunas e valores de lâmina
        lamina_columns = [f"lamina_at_{angle:03d}" for angle in range(360)]
        lamina_values = cycle_data['lamina_360'].tolist()
        
        # Contar ângulos com cobertura (não-zero)
        total_angles_covered = int(np.count_nonzero(cycle_data['lamina_360']))
        
        # Construir SQL
        base_columns = ['cycle_id', 'pivo_id', 'start_date', 'end_date', 
                       'blade_factor', 'duration_minutes', 'total_angles_covered']
        all_columns = base_columns + lamina_columns
        
        placeholders = ['%s'] * len(all_columns)
        
        insert_sql = f"""
        INSERT INTO cycle_lamina_data ({', '.join(all_columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT (cycle_id, pivo_id) 
        DO UPDATE SET
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            blade_factor = EXCLUDED.blade_factor,
            duration_minutes = EXCLUDED.duration_minutes,
            total_angles_covered = EXCLUDED.total_angles_covered,
            {', '.join([f'lamina_at_{i:03d} = EXCLUDED.lamina_at_{i:03d}' for i in range(360)])}
        """
        
        # Preparar valores
        base_values = [
            cycle_data['cycle_id'],
            cycle_data['pivo_id'],
            cycle_data['start_date'],
            cycle_data['end_date'],
            cycle_data['blade_factor'],
            cycle_data['duration_minutes'],
            total_angles_covered
        ]
        
        all_values = base_values + lamina_values
        
        try:
            self.cursor.execute(insert_sql, all_values)
            self.conn.commit()
            print(f"[OK] Ciclo inserido: {cycle_data['cycle_id']} ({cycle_data['pivo_id']})")
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao inserir: {e}")
            self.conn.rollback()
            return False

    def query_angle_history(self, pivo_id, angle, start_date=None, end_date=None):
        """
        Consulta histórico de lâmina para um ângulo específico
        
        Retorna: DataFrame com [cycle_id, start_date, end_date, lamina_value]
        """
        
        angle_column = f"lamina_at_{angle:03d}"
        
        sql = f"""
        SELECT cycle_id, pivo_id, start_date, end_date, duration_minutes,
               {angle_column} as lamina_value, blade_factor
        FROM cycle_lamina_data
        WHERE pivo_id = %s
        """
        
        params = [pivo_id]
        
        if start_date:
            sql += " AND start_date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND end_date <= %s"
            params.append(end_date)
            
        sql += " ORDER BY start_date"
        
        try:
            self.cursor.execute(sql, params)
            columns = [desc[0] for desc in self.cursor.description]
            results = self.cursor.fetchall()
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"[ERRO] Consulta falhou: {e}")
            return pd.DataFrame()

    def query_cycles_summary(self, pivo_id=None, start_date=None, end_date=None):
        """Consulta resumo de ciclos"""
        
        sql = """
        SELECT cycle_id, pivo_id, start_date, end_date,
               duration_minutes, total_angles_covered, blade_factor
        FROM cycle_lamina_data
        WHERE 1=1
        """
        
        params = []
        if pivo_id:
            sql += " AND pivo_id = %s"
            params.append(pivo_id)
        if start_date:
            sql += " AND start_date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND end_date <= %s"
            params.append(end_date)
            
        sql += " ORDER BY start_date DESC"
        
        try:
            self.cursor.execute(sql, params)
            columns = [desc[0] for desc in self.cursor.description]
            results = self.cursor.fetchall()
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"[ERRO] Consulta falhou: {e}")
            return pd.DataFrame()

    def get_cycle_full_data(self, cycle_id):
        """Recupera todos os dados de um ciclo específico (360 graus)"""
        
        lamina_columns = [f"lamina_at_{i:03d}" for i in range(360)]
        
        sql = f"""
        SELECT cycle_id, pivo_id, start_date, end_date, blade_factor,
               duration_minutes, {', '.join(lamina_columns)}
        FROM cycle_lamina_data
        WHERE cycle_id = %s
        """
        
        try:
            self.cursor.execute(sql, [cycle_id])
            result = self.cursor.fetchone()
            
            if result:
                # Extrair valores de lâmina
                lamina_values = np.array(result[6:], dtype=float)
                
                return {
                    'cycle_id': result[0],
                    'pivo_id': result[1],
                    'start_date': result[2],
                    'end_date': result[3],
                    'blade_factor': result[4],
                    'duration_minutes': result[5],
                    'lamina_360': lamina_values
                }
            return None
        except Exception as e:
            print(f"[ERRO] Consulta falhou: {e}")
            return None


# Funções auxiliares para teste
def test_database():
    """Testa a configuração do banco"""
    
    db = IrrigationDatabase(
        host='localhost',
        database='irrigation_db',
        user='postgres',
        password='admin'
    )

    if db.connect():
        db.create_tables()
        
        # Teste com dados de exemplo
        sample_lamina = np.random.rand(360) * 10
        
        sample_cycle = {
            'cycle_id': 'TEST_CYCLE_001',
            'pivo_id': 'Pivo2',
            'start_date': datetime(2025, 7, 21, 14, 30, 15),
            'end_date': datetime(2025, 7, 21, 15, 10, 45),
            'blade_factor': 5.46,
            'duration_minutes': 40,
            'lamina_360': sample_lamina
        }
        
        db.insert_cycle_data(sample_cycle)
        
        # Teste de consultas
        print("\n[TESTE] Histórico do ângulo 45°:")
        history = db.query_angle_history('Pivo2', 45)
        print(history)
        
        print("\n[TESTE] Resumo de ciclos:")
        summary = db.query_cycles_summary('Pivo2')
        print(summary)
        
        db.disconnect()

if __name__ == "__main__":
    test_database()