-- 1. Create the database
CREATE DATABASE dlproject;

-- 2. Select the database
USE dlproject;

-- 3. Create the users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- 4. Check if it worked
SHOW TABLES;
DESCRIBE users;