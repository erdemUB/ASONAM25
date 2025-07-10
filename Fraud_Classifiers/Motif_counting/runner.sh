dclist='200 400 700'


for dc in $dclist
	do
		../Motif_counting_cpp/TMC ../../Data/JPMC_Data/sim_jpmc.txt ${dc} 600 3 3 5 NO NO YES 200 300
done
