# Opacity card — sureness on the event dots

A true event is a solid dot. The model paints that dot with opacity equal to its probability. Predicted opacity 0.20 on a solid 1 is 0.20 of the way there and 0.80 short. On true is the sum of that opacity over solid dots. On false is the sum over blank dots. Precision = on true / (on true + on false). Recall = on true / solid dots, which is the average opacity on a true dot. Mean on false is the average opacity on a blank dot. The score is the harmonic mean of precision and recall, times the ratio of the smaller opacity mass to the larger of (solid dots, total opacity). No probability is thresholded.

The event marker is the first dot of the channel: solid when the player has at least one goal or assist in that gameweek. Log loss and Brier on each row are that same dot.

## Event marker

| arm | revealed | channel | log loss | Brier | solid dots | on true | on false | mean on false | precision | recall | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tversky | 20% | goals | 0.1966 | 0.0525 | 730 | 179.7 | 640.3 | 0.043 | 0.219 | 0.246 | 0.206 |
| tversky | 40% | goals | 0.1924 | 0.0519 | 544 | 138.3 | 507.4 | 0.044 | 0.214 | 0.254 | 0.196 |
| tversky | 60% | goals | 0.1929 | 0.0522 | 360 | 94.1 | 341.1 | 0.045 | 0.216 | 0.261 | 0.196 |
| tversky | 80% | goals | 0.1904 | 0.0508 | 199 | 54.4 | 171.7 | 0.042 | 0.241 | 0.273 | 0.225 |
| tversky | 20% | assists | 0.2185 | 0.0582 | 707 | 115.3 | 697.8 | 0.047 | 0.142 | 0.163 | 0.132 |
| tversky | 40% | assists | 0.2158 | 0.0578 | 531 | 89.2 | 551.0 | 0.047 | 0.139 | 0.168 | 0.126 |
| tversky | 60% | assists | 0.2124 | 0.0571 | 351 | 65.6 | 365.8 | 0.048 | 0.152 | 0.187 | 0.136 |
| tversky | 80% | assists | 0.2167 | 0.0581 | 192 | 33.1 | 191.1 | 0.047 | 0.147 | 0.172 | 0.136 |
| log_loss | 20% | goals | 0.1402 | 0.0390 | 730 | 110.8 | 582.0 | 0.039 | 0.160 | 0.152 | 0.148 |
| log_loss | 40% | goals | 0.1350 | 0.0376 | 544 | 90.7 | 479.3 | 0.041 | 0.159 | 0.167 | 0.155 |
| log_loss | 60% | goals | 0.1352 | 0.0376 | 360 | 63.9 | 332.4 | 0.043 | 0.161 | 0.177 | 0.154 |
| log_loss | 80% | goals | 0.1361 | 0.0381 | 199 | 38.0 | 184.5 | 0.045 | 0.171 | 0.191 | 0.161 |
| log_loss | 20% | assists | 0.1497 | 0.0404 | 707 | 77.2 | 595.3 | 0.040 | 0.115 | 0.109 | 0.106 |
| log_loss | 40% | assists | 0.1460 | 0.0392 | 531 | 62.6 | 478.6 | 0.041 | 0.116 | 0.118 | 0.115 |
| log_loss | 60% | assists | 0.1443 | 0.0389 | 351 | 44.7 | 324.7 | 0.042 | 0.121 | 0.127 | 0.118 |
| log_loss | 80% | assists | 0.1475 | 0.0398 | 192 | 25.1 | 179.2 | 0.044 | 0.123 | 0.131 | 0.119 |
| brier | 20% | goals | 0.1442 | 0.0392 | 730 | 106.6 | 591.6 | 0.040 | 0.153 | 0.146 | 0.143 |
| brier | 40% | goals | 0.1386 | 0.0376 | 544 | 87.1 | 487.2 | 0.042 | 0.152 | 0.160 | 0.148 |
| brier | 60% | goals | 0.1384 | 0.0377 | 360 | 60.5 | 334.2 | 0.044 | 0.153 | 0.168 | 0.146 |
| brier | 80% | goals | 0.1393 | 0.0381 | 199 | 35.6 | 182.5 | 0.045 | 0.163 | 0.179 | 0.156 |
| brier | 20% | assists | 0.1538 | 0.0403 | 707 | 73.4 | 636.0 | 0.042 | 0.103 | 0.104 | 0.103 |
| brier | 40% | assists | 0.1504 | 0.0392 | 531 | 58.0 | 507.7 | 0.044 | 0.103 | 0.109 | 0.099 |
| brier | 60% | assists | 0.1486 | 0.0389 | 351 | 40.7 | 339.9 | 0.044 | 0.107 | 0.116 | 0.103 |
| brier | 80% | assists | 0.1513 | 0.0397 | 192 | 22.5 | 183.8 | 0.045 | 0.109 | 0.117 | 0.105 |

## Every event dot

The other two dots in the channel are the unary squares for two and for three. A brace is two solid goal dots. The same sureness sums cover every square the loss trains.

| arm | revealed | channel | log loss | Brier | solid dots | on true | on false | mean on false | precision | recall | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tversky | 20% | goals | 0.0777 | 0.0191 | 805 | 180.0 | 780.2 | 0.017 | 0.187 | 0.224 | 0.171 |
| tversky | 40% | goals | 0.0756 | 0.0188 | 598 | 138.5 | 616.1 | 0.017 | 0.184 | 0.232 | 0.162 |
| tversky | 60% | goals | 0.0754 | 0.0188 | 394 | 94.2 | 412.9 | 0.017 | 0.186 | 0.239 | 0.162 |
| tversky | 80% | goals | 0.0753 | 0.0185 | 219 | 54.4 | 210.1 | 0.017 | 0.206 | 0.249 | 0.186 |
| tversky | 20% | assists | 0.0812 | 0.0203 | 749 | 115.4 | 845.1 | 0.018 | 0.120 | 0.154 | 0.105 |
| tversky | 40% | assists | 0.0794 | 0.0200 | 558 | 89.2 | 665.4 | 0.019 | 0.118 | 0.160 | 0.101 |
| tversky | 60% | assists | 0.0790 | 0.0199 | 372 | 65.7 | 441.4 | 0.019 | 0.130 | 0.177 | 0.110 |
| tversky | 80% | assists | 0.0808 | 0.0203 | 204 | 33.1 | 231.6 | 0.018 | 0.125 | 0.162 | 0.109 |
| log_loss | 20% | goals | 0.0554 | 0.0146 | 805 | 112.7 | 673.1 | 0.015 | 0.143 | 0.140 | 0.138 |
| log_loss | 40% | goals | 0.0531 | 0.0140 | 598 | 92.2 | 555.9 | 0.016 | 0.142 | 0.154 | 0.137 |
| log_loss | 60% | goals | 0.0531 | 0.0139 | 394 | 64.8 | 386.6 | 0.016 | 0.144 | 0.164 | 0.134 |
| log_loss | 80% | goals | 0.0539 | 0.0142 | 219 | 38.6 | 214.9 | 0.017 | 0.152 | 0.176 | 0.141 |
| log_loss | 20% | assists | 0.0559 | 0.0143 | 749 | 77.7 | 672.2 | 0.015 | 0.104 | 0.104 | 0.104 |
| log_loss | 40% | assists | 0.0538 | 0.0138 | 558 | 62.9 | 542.1 | 0.015 | 0.104 | 0.113 | 0.100 |
| log_loss | 60% | assists | 0.0538 | 0.0138 | 372 | 45.0 | 368.9 | 0.016 | 0.109 | 0.121 | 0.103 |
| log_loss | 80% | assists | 0.0552 | 0.0142 | 204 | 25.3 | 203.7 | 0.016 | 0.111 | 0.124 | 0.104 |
| brier | 20% | goals | 0.0591 | 0.0146 | 805 | 108.6 | 816.5 | 0.018 | 0.117 | 0.135 | 0.109 |
| brier | 40% | goals | 0.0568 | 0.0140 | 598 | 88.7 | 666.6 | 0.019 | 0.117 | 0.148 | 0.104 |
| brier | 60% | goals | 0.0566 | 0.0140 | 394 | 61.4 | 455.6 | 0.019 | 0.119 | 0.156 | 0.103 |
| brier | 80% | goals | 0.0574 | 0.0142 | 219 | 36.2 | 248.2 | 0.020 | 0.127 | 0.165 | 0.111 |
| brier | 20% | assists | 0.0598 | 0.0143 | 749 | 73.9 | 844.4 | 0.018 | 0.080 | 0.099 | 0.072 |
| brier | 40% | assists | 0.0580 | 0.0138 | 558 | 58.4 | 672.6 | 0.019 | 0.080 | 0.105 | 0.069 |
| brier | 60% | assists | 0.0580 | 0.0139 | 372 | 41.0 | 450.8 | 0.019 | 0.083 | 0.110 | 0.072 |
| brier | 80% | assists | 0.0591 | 0.0142 | 204 | 22.7 | 243.6 | 0.019 | 0.085 | 0.111 | 0.074 |

## Call

Event-marker means. Tversky precision **0.184**, recall **0.216**. Log loss precision **0.141**, recall **0.147**. Brier precision **0.130**, recall **0.137**. Training loss on every remainder event bit: log loss **0.0543**, Brier arm **0.0141**.
