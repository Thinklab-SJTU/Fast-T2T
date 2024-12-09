start_time=$SECONDS
python -u main.py solve \
    kamis \
    ../mis/snap_processed/facebook_combined \
    ../mis/snap_processed/facebook_combined_label \
    --time_limit 600
end_time=$SECONDS
echo "Total execution time : ${end_time} - ${start_time} seconds."