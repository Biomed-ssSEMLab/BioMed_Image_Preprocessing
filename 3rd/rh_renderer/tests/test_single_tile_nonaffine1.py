# Takes a single tile and applies some transformations on it
from __future__ import print_function
import pylab
from rh_renderer.single_tile_renderer import SingleTileRenderer
from rh_renderer import models
import numpy as np

if __name__ == '__main__':
    tile_fname = 'images/tile1.bmp'
    renderer = SingleTileRenderer(tile_fname, 3348, 2976, compute_mask=False, compute_distances=False)

    img, start_point = renderer.render()
    print("Before transformations: Start point is at:", start_point, "image shape:", img.shape)
    pylab.figure()
    pylab.imshow(img, cmap='gray', vmin=0., vmax=255.)

    transform1 = models.RigidModel()
    transform1.set_from_modelspec('0.000385855181988 15014.2735713 11052.6315792')

    transform2 = models.PointsTransformModel()
    transform2.set_from_modelspec('14055.9983674 11019.9985055 6370.8177883 13052.1242724 1.0 14330.9983674 10469.9985055 6374.13943569 12464.339873 1.0 14605.9983674 11019.9985055 6858.96020641 12797.8440539 1.0 14880.9983674 10469.9985055 6861.65601032 12210.2389664 1.0 15155.9983674 11019.9985055 7346.36989866 12543.2328822 1.0 15430.9983674 10469.9985055 7348.80203199 11955.3414738 1.0 15705.9983674 11019.9985055 7833.50408248 12288.3449961 1.0 15980.9983674 10469.9985055 7836.34936775 11700.1025955 1.0 16255.9983674 11019.9985055 8321.19263605 12033.9185328 1.0 16530.9983674 10469.9985055 8323.53276644 11445.5466863 1.0 16805.9983674 11019.9985055 8809.17662011 11779.8585746 1.0 17080.9983674 10469.9985055 8812.78115494 11191.648332 1.0 14055.9983674 12119.9985055 6852.92148361 13972.0753324 1.0 14330.9983674 11569.9985055 6855.01422636 13384.5343354 1.0 14605.9983674 12119.9985055 7340.46858607 13716.9663191 1.0 14330.9983674 12669.9985055 7338.34873165 14306.4489043 1.0 14880.9983674 11569.9985055 7343.29903359 13130.3195134 1.0 14880.9983674 12669.9985055 7825.91324188 14051.1294982 1.0 14055.9983674 13219.9985055 7336.81515619 14895.477592 1.0 14055.9983674 14319.9985055 7821.08599898 15817.2553387 1.0 14605.9983674 13219.9985055 7824.15294375 14640.5340601 1.0 14330.9983674 13769.9985055 7822.61002661 15227.9345715 1.0 14605.9983674 14319.9985055 8308.82799697 15562.2624493 1.0 14880.9983674 13769.9985055 8310.16203238 14973.4381267 1.0 15155.9983674 12119.9985055 7828.64822078 13461.4066096 1.0 15430.9983674 11569.9985055 7831.02826158 12875.1100429 1.0 15705.9983674 12119.9985055 8316.8793411 13207.180783 1.0 15430.9983674 12669.9985055 8314.40815509 13796.8179187 1.0 15980.9983674 11569.9985055 8318.39605191 12620.2822162 1.0 15980.9983674 12669.9985055 8802.17028515 13540.5812961 1.0 16255.9983674 12119.9985055 8804.05642873 12951.5459326 1.0 16530.9983674 11569.9985055 8806.18140328 12366.1500808 1.0 16805.9983674 12119.9985055 9291.85152868 12697.0539681 1.0 16530.9983674 12669.9985055 9289.43728062 13285.4543118 1.0 17080.9983674 11569.9985055 9294.27014506 12112.3056335 1.0 17080.9983674 12669.9985055 9777.39362222 13030.7339729 1.0 15155.9983674 13219.9985055 8312.3448769 14385.8663678 1.0 15155.9983674 14319.9985055 8796.52299457 15307.4140321 1.0 15705.9983674 13219.9985055 8800.63118261 14130.2883606 1.0 15430.9983674 13769.9985055 8798.53459643 14718.5089331 1.0 15705.9983674 14319.9985055 9283.93836313 15052.0690478 1.0 15980.9983674 13769.9985055 9286.19600935 14462.7282345 1.0 16255.9983674 13219.9985055 9288.06144338 13874.773653 1.0 16255.9983674 14319.9985055 9770.16478031 14795.8039995 1.0 16805.9983674 13219.9985055 9775.14748224 13619.2476379 1.0 16530.9983674 13769.9985055 9773.51620368 14206.89926 1.0 16805.9983674 14319.9985055 10258.0274231 14540.3206696 1.0 17080.9983674 13769.9985055 10260.8713084 13951.4022307 1.0 17355.9983674 11019.9985055 9298.50644389 11526.7469532 1.0 17630.9983674 10469.9985055 9301.76867378 10937.2056687 1.0 17905.9983674 11019.9985055 9786.92776129 11272.1934928 1.0 18180.9983674 10469.9985055 9791.55168773 10683.4716308 1.0 18455.9983674 11019.9985055 10275.0200512 11017.2975536 1.0 18730.9983674 10469.9985055 10278.4321284 10429.1388993 1.0 19005.9983674 11019.9985055 10763.3339438 10763.0354394 1.0 19280.9983674 10469.9985055 10765.4617324 10174.7976799 1.0 17355.9983674 12119.9985055 9780.22459895 12442.6741209 1.0 17630.9983674 11569.9985055 9783.39241511 11858.055423 1.0 17905.9983674 12119.9985055 10268.7781786 12188.8636123 1.0 17630.9983674 12669.9985055 10265.9482613 12776.6805571 1.0 18180.9983674 11569.9985055 10271.9186855 11603.4892994 1.0 18455.9983674 12119.9985055 10757.3350302 11934.9343551 1.0 18180.9983674 12669.9985055 10754.3126092 12522.3845387 1.0 18730.9983674 11569.9985055 10760.6831455 11349.5528275 1.0 19005.9983674 12119.9985055 11246.1735702 11680.7561696 1.0 18730.9983674 12669.9985055 11243.3272243 12267.3613525 1.0 17355.9983674 13219.9985055 10263.4054227 13364.0564105 1.0 17355.9983674 14319.9985055 10746.4074456 14284.6276141 1.0 17905.9983674 13219.9985055 10752.1958634 13109.2168981 1.0 17630.9983674 13769.9985055 10748.8359264 13695.8512252 1.0 17905.9983674 14319.9985055 11234.1857292 14029.5007663 1.0 18180.9983674 13769.9985055 11237.3934992 13441.0984044 1.0 18455.9983674 13219.9985055 11241.0728425 12854.4960772 1.0 18455.9983674 14319.9985055 11722.1525729 13774.8513207 1.0 19005.9983674 13219.9985055 11729.8208453 12600.3530675 1.0 18730.9983674 13769.9985055 11726.1852218 13186.7209697 1.0 19005.9983674 14319.9985055 12209.8390562 13519.7636152 1.0 19280.9983674 11569.9985055 11248.8826562 11096.2517072 1.0 19280.9983674 12669.9985055 11731.5771226 12013.4276159 1.0 19280.9983674 13769.9985055 12213.8615547 12932.3762663 1.0 14330.9983674 14869.9985055 8307.44931244 16152.4955001 1.0 14880.9983674 14869.9985055 8795.29711773 15897.9213803 1.0 15430.9983674 14869.9985055 9283.15140039 15642.4496628 1.0 15980.9983674 14869.9985055 9769.92684555 15386.7024332 1.0 16530.9983674 14869.9985055 10255.7612955 15130.1516657 1.0 17080.9983674 14869.9985055 10744.0774546 14874.195028 1.0 17630.9983674 14869.9985055 11232.5286542 14618.8101981 1.0 18180.9983674 14869.9985055 11720.0539757 14363.6448448 1.0 18730.9983674 14869.9985055 12207.7525278 14108.1030419 1.0 19280.9983674 14869.9985055 12695.1148827 13852.6696047 1.0')

    transform3 = models.TranslationModel()
    transform3.set_from_modelspec('-7000.0 -7700.0')


    # Add the transformations and render the result
    transforms = [transform1, transform2, transform3] 
    for t in transforms:
        print("Adding transformation:", t)
        renderer.add_transformation(t)

    img, start_point = renderer.render()

    print("Start point is at:", start_point, "image shape:", img.shape)
    pylab.figure()
    pylab.imshow(img, cmap='gray', vmin=0., vmax=255.)



    pylab.show()

