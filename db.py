import sqlite3

DB_PATH = "sneakers.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sneakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            brand TEXT,
            color TEXT,
            description TEXT,
            annoy_id INTEGER UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            sneaker_id INTEGER,
            action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_sneaker(filename, brand, color, description, annoy_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sneakers (filename, brand, color, description, annoy_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, brand, color, description, annoy_id))
    conn.commit()
    conn.close()

def get_sneaker_by_annoy_id(annoy_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM sneakers WHERE annoy_id = ?", (annoy_id,))
    result = c.fetchone()
    conn.close()
    return result
