import mysql.connector

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': ''
}

try:
    # 1. Connect to MySQL Server (without DB)
    print("Connecting to MySQL...")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 2. Create Database
    print("Creating database 'dlproject'...")
    cursor.execute("CREATE DATABASE IF NOT EXISTS dlproject")
    
    # 3. Select Database
    cursor.execute("USE dlproject")

    # 4. Create Users Table (with Email column)
    print("Creating 'users' table...")
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(255) NOT NULL UNIQUE,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        password VARCHAR(255) NOT NULL
                    )''')
    
    print("Database and Table setup complete successfully!")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()
