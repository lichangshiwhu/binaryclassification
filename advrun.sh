




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
python advmain.py -F ./scriptstest/test_adv_ShellsorPebbles_alpha.yml -O test_adv_ShellsorPebbles_alpha_sig -B 0 -E 3 
python advmain.py -F ./scriptstest/test_adv_catanddog_alpha.yml -O test_adv_catanddog_alpha_sig -B 0 -E 3 
python advmain.py -F ./scriptstest/test_adv_CactusAerialPhotos_alpha.yml -O test_adv_CactusAerialPhotos_alpha_sig -B 0 -E 3
