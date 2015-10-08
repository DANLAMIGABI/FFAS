from openmdao.lib.casehandlers.api import CaseDataset
import csv
#----------------------------------------------------
# Print out history of our objective for inspection
#----------------------------------------------------
case_dataset = CaseDataset('opt_record.json', 'json')
data = case_dataset.data.by_case().fetch()

csvfile = open('opt_record.csv', 'wb')
spamwriter = csv.writer(csvfile)
for case in data:
	comp = case['compatibility.compatibility']
	print comp
	if comp == 0:
		if 'driver.gen_num' in case: gen = [case['driver.gen_num']]
		else:						 gen = [None]	
		in_opts = [case['compatibility.option1_out'], case['compatibility.option2_out'], case['compatibility.option3_out'], 
				   case['compatibility.option4_out'], case['compatibility.option5_out'], case['compatibility.option6_out'], 
				   case['compatibility.option7_out'], case['compatibility.option8_out'], case['compatibility.option9_out'], ]

		spamwriter.writerow(gen+in_opts)

#spamwriter.close()

