#!/usr/bin/expect

# to set RAINIER_PATH, execute the following in terminal, or save in .bashrc/profile
# export RAINIER_PATH=/path/to/RAINIER

if { [info exists RAINIER_PATH] } {
	send "\nRAINIER_PATH is unset!\n"
    } else {

	   	# if settings.h exisits in the RAINIER folder, this will be read by RAINIER instead.
		if {[file exists $env(RAINIER_PATH)/settings.h]} {
   		send "Cannot run -- settings.h exists also in RAINIER folder\n"
		} else {
		spawn root -l 

		set timeout -1

		send "dir = \".include \" + gSystem->GetWorkingDirectory()\r"
		send "gROOT->ProcessLine(dir.c_str())\r"
		expect -timeout -1 "root" { send ".x $env(RAINIER_PATH)/RAINIER.C++\r"}
		# send "gROOT->ProcessLine(\".L $env(RAINIER_PATH)/Analyze.C++\")\r"
		send ".L $env(RAINIER_PATH)/Analyze.C++\r"
		send "RetrievePars()\r"
		interact
		}
	}