#from pydoc import classname
import sqlite3
import os 

class dbman: 
    def __init__(self,dbpath):
        self.dbpath = dbpath 
        self.conn = sqlite3.connect(dbpath)
        self.createTable()

    def createTable(self):
        #creates a database to store file paths and metadata
        query = '''
                CREATE TABLE IF NOT EXISTS spectrograms(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opath TEXT NOT NULL, 
                spath TEXT NOT NULL, 
                classname TEXT NOT NULL, 
                pdate TEXT
            )'''
        self.conn.execute(query)
        self.conn.commit()
    def addSpectrogramEntry(self,opath,spath,classname,pdate):
        #checks to see if the entry already exists in the database
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM spectrograms WHERE opath = ?', (opath,))
        result = cursor.fetchone()
        if result is None: 
             #insert new entry into database
             query = 'INSERT INTO spectrograms(opath,spath,classname,pdate) VALUES(?,?,?,?)'
             cursor.execute(query,(opath,spath,classname,pdate))
        else:
            #update entries that already exist 
            query = 'UPDATE spectrograms SET spath = ?, classname = ?,  pdate = ? WHERE opath = ?'
            cursor.execute(query,(spath,pdate,classname,opath))
        self.conn.commit()

    def close(self):
        self.conn.close()