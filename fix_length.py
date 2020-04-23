#coding:utf-8
bed_file = open("HepG2_POLR2A.bed")
fixed_length_bed_file = open("HepG2_POLR2A_fixed.bed","wr")
#fixed_length_bed_file.write("#coding:utf-8\n")
#fixed_length_bed_file.write("chrom\t" + "start\t" + "end\n")

for line in bed_file.readlines():
	line_content = line.split()
	chrom_index = line_content[0]
	peak_start = int(line_content[1])
	peak_end   = int(line_content[2])
	peak_middle = (peak_end + peak_start) // 2
	fixed_length_peak_start = peak_middle - 150
	fixed_length_peak_end   = peak_middle + 150
	if fixed_length_peak_start < 0:
		continue
	fixed_length_bed_file.write(chrom_index + "\t" + str(fixed_length_peak_start) + "\t"
								+ str(fixed_length_peak_end) + "\n")