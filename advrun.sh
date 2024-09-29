python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles.yml -O test_adv_ShellsorPebbles -B 0 -E 3 &
python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles.yml -O test_adv_Sop_sigmoid -B 0 -E 3 &
python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_alpha.yml -O test_adv_ShellsorPebbles_alpha -B 0 -E 3 &

python advmain.py -F ./scriptstest/test_adv_catanddog.yml -O test_adv_catanddog -B 0 -E 3 &
python advmain.py -F ./scriptstest/test_adv_catanddog.yml -O test_adv_catanddog_sigmoid -B 0 -E 3 &
python advmain.py -F ./scriptstest/test_adv_catanddog_alpha.yml -O test_adv_catanddog_alpha -B 0 -E 3 &


python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos.yml -O test_adv_cactusaerialphotos_log -B 0 -E 3 &
python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos_alpha.yml -O test_adv_CactusAerialPhotos_alpha -B 0 -E 3 &

python advmain.py -F ./scriptstest/test_adv_makecircle.yml -O test_adv_makecircle_num -B 0 -E 5 &
python advmain.py -F ./scriptstest/test_adv_makeMoon.yml -O test_adv_makeMoon_num -B 0 -E 5 &


# sigmoid loss under alpha
python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos_epsilon.yml -O test_adv_CactusAerialPhotos_epsilon -B 0 -E 3 ; python advmain.py -F ./scriptstest/test_adv_catanddog_epsilon.yml -O test_adv_catanddog_epsilon -B 0 -E 3  ; python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_epsilon.yml -O test_adv_ShellsorPebbles_epsilon -B 0 -E 3

python advmain.py -F ./scriptstest/test_adv_catanddog_epsilon.yml -O test_adv_catanddog_epsilon_sig -B 0 -E 3  ; python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_epsilon.yml -O test_adv_ShellsorPebbles_epsilon_sig -B 0 -E 3

# sigmoidloss
# python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos.yml -O test_adv_cactusaerialphotos_sig_0p1_alpha -B 1 -E 3 ; python advmain.py -F ./scriptstest/test_adv_catanddog.yml -O test_adv_catanddog_sig_0p1_alpha -B 0 -E 3 ; python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles.yml -O test_adv_ShellsorPebbles_sig_0p1_alpha -B 0 -E 3

# toy test for margin
# python advmain.py -F ./scriptstest/test_adv_makeCircleX2Y2_log_margin.yml -O test_adv_makeCircleX2Y2_log_margin -B 0 -E 5 ; python advmain.py -F ./scriptstest/test_adv_makeCircleX2Y2_sig_margin.yml -O test_adv_makeCircleX2Y2_sig_margin -B 0 -E 5 

python advmain.py -F ./scriptstest/test_adv_makeCircleX2Y2_margin.yml -O test_adv_makeCircleX2Y2_margin -B 0 -E 5

python advmain.py -F ./scriptstest/test_adv_makeByYeqX_margin.yml -O test_adv_makeByYeqX_margin_nn -B 0 -E 10

python advmain.py -F ./scriptstest/test_adv_catanddog_epsilon.yml -O test_adv_catanddog_epsilon_baseline_iter30 -B 0 -E 3  ; python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_epsilon.yml -O test_adv_ShellsorPebbles_epsilon_baseline_iter30 -B 0 -E 3 ; python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos_epsilon.yml -O test_adv_CactusAerialPhotos_epsilon_baseline_iter30 -B 0 -E 3

python advmain.py -F ./scriptstest/test_adv_makeByYeqX_margin.yml -O test_adv_makeByYeqX_margin_sovr -B 0 -E 10

python advmain.py -F ./scriptstest/test_adv_catanddog_epsilon.yml -O test_adv_catanddog_epsilon_baseline_mail_sovr -B 0 -E 3  ; python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_epsilon.yml -O test_adv_ShellsorPebbles_epsilon_baseline_mail_sovr -B 0 -E 3 ; python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos_epsilon.yml -O test_adv_CactusAerialPhotos_epsilon_baseline_mail_sovr -B 0 -E 3

