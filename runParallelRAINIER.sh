#!/usr/bin/expect

# requires expect language
# to set RAINIER_PATH, execute the following in terminal, or save in .bashrc/profile
# export RAINIER_PATH=/path/to/RAINIER

if { [info exists RAINIER_PATH] } {
	send "\nRAINIER_PATH is unset!\n"
    } else {    
        file copy -force $env(RAINIER_PATH)/RAINIER.C RAINIER_copy.C
		spawn root -l

		set timeout -1

		send "gSystem->SetIncludePath(\"-fopenmp\")\r"
		send "gSystem->AddLinkedLibs(\"-lgomp\")\r"
		expect -timeout -1 "root" { send ".x RAINIER_copy.C++\r"}
		interact
}