import ftplib

url = 'ftp.horizon-systems.com'

ftp = ftplib.FTP(url,'USGS_P133','ws54R7')

print ftp.nlst()

ftp.quit()