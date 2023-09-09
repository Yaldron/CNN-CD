load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

; Do the bootstrap test and return the prob
function Boot(X[*][*][*]:double,Y[*][*][*]:double,siglvl[*]:float)
local BootStrap,diffBoot,diffBootLow,diffBootHi,diff,prob
begin
BootStrap   = bootstrap_diff(X,Y, 2000, (/0,0/), False)
diffBoot    = BootStrap[0]
diffBootLow = bootstrap_estimate(diffBoot, siglvl*2.0, False)
diffBootHi  = bootstrap_estimate(diffBoot, 1-siglvl*2.0, False) 
diff        = dim_avg_n(X,0)-dim_avg_n(Y, 0)
prob        = where(sign_matlab(diff).eq.sign_matlab(diffBootHi).and.\
                      sign_matlab(diff).eq.sign_matlab(diffBootLow),99.,50.)
copy_VarCoords(X(0,:,:), diff)
copy_VarCoords(X(0,:,:), prob)
return prob
end


begin
res = True 
res@gsnDraw              = False
res@gsnFrame             = False
res@tmXTOn               = False
res@tmYROn               = False

res@cnLineThicknessF             = 2.0
res@gsnContourZeroLineThicknessF = 3.0
res@gsnContourNegLineDashPattern =  1
res@cnInfoLabelOn        = False
res@cnLineLabelsOn       = False


res2                     = res
res2@cnLineColor         = "red"


res@gsnAddCyclic         = True
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels             = (/-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4/)



;-------------------------------------------------------------------------
tres                       = True              
tres@gsnDraw               = False             
tres@gsnFrame              = False            
tres@cnMonoFillPattern     = False
tres@cnMonoFillColor       = True
tres@cnLinesOn             = False
tres@cnLineLabelsOn        = False
tres@cnInfoLabelOn         = False
tres@lbLabelBarOn          = False
tres@cnLevelSelectionMode  = "ExplicitLevels"  
tres@cnLevels              = (/95/)            
tres@cnFillOn              = True
tres@gsnSpreadColors       = False
tres@cnFillPatterns        = (/-1,8/)
tres@cnConstFEnableFill    = True
;tres@cnFillColor           = "firebrick1"
tres@cnFillColor           = "black"


tres2                       = True              
tres2@gsnDraw               = False             
tres2@gsnFrame              = False            
tres2@cnMonoFillPattern     = False
tres2@cnMonoFillColor       = True
tres2@cnLinesOn             = False
tres2@cnLineLabelsOn        = False
tres2@cnInfoLabelOn         = False
tres2@lbLabelBarOn          = False
tres2@cnLevelSelectionMode  = "ExplicitLevels"  
tres2@cnLevels              = (/95/)            
tres2@cnFillOn              = True
tres2@gsnSpreadColors       = False
tres2@cnFillPatterns        = (/-1,17/)
tres2@cnConstFEnableFill    = True
tres2@cnFillDotSizeF        = 0.0025

wks = gsn_open_wks("eps", "KeyRegions_CNN-CD_10-month-lead")
plot0 = new(2,graphic)
plot1 = new(2,graphic)
plot2 = new(2,graphic)
plot3 = new(2,graphic)
plot4 = new(2,graphic)

;-------------------------------------------------------------------------

Mres = True
Mres@cnFillOn   = True
Mres@cnFillMode = "RasterFill"
Mres@cnLinesOn  = False
Mres@mpFillDrawOrder  = "PostDraw"

Mres@cnLevelSelectionMode = "ManualLevels" ; set manual contour levels
Mres@cnMinLevelValF       = -0.25               ; set min contour level
Mres@cnMaxLevelValF       =  0.25               ; set max contour level
Mres@cnLevelSpacingF      =  0.05               ; set contour spacing

Mres@cnLineLabelsOn       = False


Mres@mpMinLonF            = 20.
Mres@mpMaxLonF            = 380.
Mres@mpCenterLonF         = 200.
Mres@mpMinLatF            = -55.
Mres@mpMaxLatF            =  60.



Mres@mpShapeMode          = "FreeAspect"
Mres@vpHeightF            = 0.5
Mres@vpWidthF             = 0.85

Mres@cnMissingValFillColor = "White"

Mres@lbLabelBarOn         = False

Mres@tmXTOn               = False
Mres@tmYROn               = False

Mres@gsnFrame             = False
Mres@gsnDraw              = False

Mres@tmBorderThicknessF = 3.0
Mres@tmXBMajorThicknessF = 3.0
Mres@tmXBMinorThicknessF = 3.0


Mres@tmYLMajorThicknessF = 3.0
Mres@tmYLMinorThicknessF = 3.0

Mres@tmXBLabelFontHeightF    = 0.022
Mres@tmXBLabelFontThicknessF = 2.0
Mres@tmYLLabelFontHeightF    = 0.022
Mres@tmXBLabelFontThicknessF = 2.0

Mres@gsnSpreadColorStart = 40
Mres@gsnSpreadColorEnd   = 210



Mres@tmXBMode        = "Explicit"
Mres@tmXBValues      = ispan(60,360,60)
Mres@tmXBMinorValues = ispan(20,380,10)
Mres@tmXBLabels      = (/"60E","120E","180","120W","60W","0"/)

Mres@gsnStringFontHeightF = 0.022
Mres@gsnLeftStringOrthogonalPosF = 0.01


;===========bootstrap test of heat values==============
fheat = addfile("../../data/HeatValue/HeatValue_04st_lead10_tarDJF.nc","r")
heat  = dim_avg_n_Wrap(fheat->HeatValue(:,4:,:,:),1)

lat = heat&lat
lon = heat&lon

lat@units = "degrees_north"
lon@units = "degrees_east"

heat&lat = lat
heat&lon = lon

ENs      = (/1982,1986,1991,1994,1997,2002,2004,2009,2014,2015/)       ;the successful predicted ENs at 10-month-lead
LNs      = (/1983,1984,1988,1998,1999,2000,2005,2007,2008,2010,2011/)  ;the successful predicted LNs at 10-month-lead

siglvl  = 0.05
prob_H1 = Boot(heat(ENs-1981,:,:),heat(:,:,:),siglvl)
prob_H2 = Boot(heat(LNs-1981,:,:),heat(:,:,:),siglvl)

Slice1 = dim_avg_n_Wrap(heat(ENs-1981,:,:),0)
Slice2 = dim_avg_n_Wrap(heat(LNs-1981,:,:),0)

Slice1 = where(Slice1.gt.0.0,Slice1,0.0)
Slice2 = where(Slice2.lt.0.0,Slice2,0.0)

Slice1 = where(prob_H1.gt.95.0,Slice1,0.0)
Slice2 = where(prob_H2.gt.95.0,Slice2,0.0)

Mres@gsnStringFontHeightF = 0.022
Mres@gsnLeftString = "(a) Key regions for CNN-CD ENs forecasts (10-month lead)"
plot0(0) = gsn_csm_contour_map(wks,Slice1, Mres)
Mres@gsnLeftString = "(b) Key regions for CNN-CD LNs forecasts (10-month lead)"
plot0(1) = gsn_csm_contour_map(wks,Slice2, Mres)


;-------------------------------------------------------------------------
f1  = addfile("../../data/Valid/anom_detrend_ersstv5_198001-201812_Ham-LR.nc", "r")
f2  = addfile("../../data/Valid/anom_detrend_sshg_GODAS_198001-201812_Ham-LR.nc", "r")

sst = f1->tos
zos = f2->zos

Ref = 1980

sst_JFM  = month_to_season(sst, "JFM")
zos_JFM  = month_to_season(zos, "JFM")

res@tmXBLabelFont = "times-bold"
res@tmYLLabelFont = "times-bold"
res@gsnStringFont = "times-bold"


GP_list   = [/ENs-Ref,LNs-Ref/]

do i = 0,1
    GP   = GP_list[i]

    BootStrap1   = bootstrap_diff(sst_JFM(GP,:,:), sst_JFM(:,:,:), 2000, (/0,0/), False)
    diffBoot1    = BootStrap1[0]
    diffBootLow1 = bootstrap_estimate(diffBoot1, siglvl*2.0, False)
    diffBootHi1  = bootstrap_estimate(diffBoot1, 1-siglvl*2.0, False) 
    diff1        = dim_avg_n(sst_JFM(GP,:,:),0)-dim_avg_n(sst_JFM(:,:,:), 0)
    prob1        = where(sign_matlab(diff1).eq.sign_matlab(diffBootHi1).and.\
                          sign_matlab(diff1).eq.sign_matlab(diffBootLow1),99.,50.)

    copy_VarCoords(sst_JFM(0,:,:), diff1)
    copy_VarCoords(sst_JFM(0,:,:), prob1)

    BootStrap2   = bootstrap_diff(zos_JFM(GP,:,:), zos_JFM(:,:,:), 2000, (/0,0/), False)
    diffBoot2    = BootStrap1[0]
    diffBootLow2 = bootstrap_estimate(diffBoot2, siglvl*2.0, False)
    diffBootHi2  = bootstrap_estimate(diffBoot2, 1-siglvl*2.0, False) 
    diff2        = dim_avg_n(zos_JFM(GP,:,:),0)-dim_avg_n(zos_JFM(:,:,:), 0)
    prob2        = where(sign_matlab(diff2).eq.sign_matlab(diffBootHi2).and.\
                          sign_matlab(diff2).eq.sign_matlab(diffBootLow2),99.,50.)


    copy_VarCoords(zos_JFM(0,:,:), diff2)
    copy_VarCoords(zos_JFM(0,:,:), prob2)

    ;draw SSTA with siglvl marked
    diff2    = where(prob2.gt.95.0, diff2, diff2@_FillValue)

    plot1(i) = gsn_csm_contour(wks, diff1, res)
    plot2(i) = gsn_csm_contour(wks, prob1, tres)
    diffZP   = dim_avg_n(zos_JFM(GP,:,:),0)-dim_avg_n(zos_JFM(:,:,:), 0)
    copy_VarCoords(zos_JFM(0,:,:), diffZP)
 
    ;draw significant SSHA  
    FLG1 = prob2
    FLG1 = where(diffZP.gt.0.0,FLG1,0.0)
    FLG2 = prob2
    FLG2 = where(diffZP.lt.0.0,FLG2,0.0)

    tres2@cnFillColor           = "red"
    plot3(i) = gsn_csm_contour(wks, FLG1, tres2)
    tres2@cnFillColor           = "darkgreen"
    plot4(i) = gsn_csm_contour(wks, FLG2, tres2)


    overlay(plot0(i),plot1(i))
    overlay(plot0(i),plot2(i))
    overlay(plot0(i),plot3(i))
    overlay(plot0(i),plot4(i))
    delete(GP)
end do

delete(res@cnLevels)


pres = True
pres@gsnMaximize      = True
pres@gsnStringFont    = "times-bold"
pres@lbLabelFont      = "times-bold"
pres@gsnPanelLabelBar = True
pres@lbBoxLinesOn     = False
pres@lbLabelFontHeightF = 0.0135
pres@lbBoxEndCapStyle = "TriangleBothEnds"
pres@gsnPanelYWhiteSpacePercent = 2.5
pres@pmLabelBarWidthF = 0.5
gsn_panel(wks, plot0, (/2,1/), pres)


end

