** This do-file is designed to set initial path.

clear all
set more off
cap log close
global sysdate = c(current_date)


if "`c(username)'" == "ttyang" {
	
	global do = "D:\nest\Dropbox\RA_research\China_Xi\data\do"
    global rdata = "D:\nest\Dropbox\RA_research\China_Xi\data\rdata"
	global wdata = "D:\nest\Dropbox\RA_research\China_Xi\data\wdata"
    
}
if "`c(username)'" == "avis1" {
    
	global do = "C:\Users\avis1\Desktop\master_thesis\do"
    global rdata = "C:\Users\avis1\Desktop\master_thesis\rdata"
	global wdata = "C:\Users\avis1\Desktop\master_thesis\wdata"
	global pic = "C:\Users\avis1\Desktop\master_thesis\pic"
	global table = "C:\Users\avis1\Desktop\master_thesis\table"
	global log = "C:\Users\avis1\Desktop\master_thesis\log"
	
}
if "`c(username)'" == "" {
    
    global do = ""
    global rdata = ""
	global wdata = ""
	
}

if "`c(username)'" == "" {

    global do = ""
    global rdata = ""
	global wdata = ""

	
}

