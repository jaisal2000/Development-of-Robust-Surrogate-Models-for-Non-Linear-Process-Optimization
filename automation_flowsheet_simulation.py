

pip install pythonnet

# remove the following two lines to run on linux
import pythoncom
pythoncom.CoInitialize()

import clr
import numpy as np
import pandas as pd

from System.IO import Directory, Path, File
from System import String, Environment

dwsimpath = "C:\\Users\\LOQ\\AppData\\Local\\DWSIM\\" # Make sure this path is correct

clr.AddReference(dwsimpath + "CapeOpen.dll")
clr.AddReference(dwsimpath + "DWSIM.Automation.dll")
clr.AddReference(dwsimpath + "DWSIM.Interfaces.dll")
clr.AddReference(dwsimpath + "DWSIM.GlobalSettings.dll")
clr.AddReference(dwsimpath + "DWSIM.SharedClasses.dll")
clr.AddReference(dwsimpath + "DWSIM.Thermodynamics.dll")
clr.AddReference(dwsimpath + "DWSIM.UnitOperations.dll")
clr.AddReference(dwsimpath + "DWSIM.Inspector.dll")
clr.AddReference(dwsimpath + "System.Buffers.dll")
clr.AddReference(dwsimpath + "DWSIM.Thermodynamics.ThermoC.dll")

from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
from DWSIM.Thermodynamics import Streams, PropertyPackages
from DWSIM.UnitOperations import UnitOperations
from DWSIM.Automation import Automation3
from DWSIM.GlobalSettings import Settings
from System import Array

Directory.SetCurrentDirectory(dwsimpath)

# --- Simulation Setup (mostly from your original code) ---
manager = Automation3()
myflowsheet = manager.CreateFlowsheet()
cnames = ["Water", "Ethanol"]
myflowsheet.AddCompound("Water")
myflowsheet.AddCompound("Ethanol")

feed  = myflowsheet.AddFlowsheetObject("Material Stream", "Feed")
dist = myflowsheet.AddFlowsheetObject("Material Stream", "Distillate")
bottoms = myflowsheet.AddFlowsheetObject("Material Stream", "Bottoms")
column = myflowsheet.AddFlowsheetObject("Distillation Column", "Column")

feed = feed.GetAsObject()
dist = dist.GetAsObject()
bottoms = bottoms.GetAsObject()
column = column.GetAsObject()

# Fixed column configurations
feed_stage = 6
column.ConnectFeed(feed, feed_stage)
column.ConnectDistillate(dist)
column.ConnectBottoms(bottoms)
myflowsheet.NaturalLayout()

feed.SetTemperature(350.0) # K
feed.SetPressure(101325.0) # Pa (1 atm)

nrtl = myflowsheet.CreateAndAddPropertyPackage("NRTL")

# --- Define Base Case and Ranges for Variation ---
base_F = 300.0 # mol/s
base_N = 12 # Number of stages

# Parameters to vary and their ranges
# Using np.linspace for smooth ranges, and np.random.choice for discrete/random selection if needed
R_values = np.linspace(0.8, 5.0, 510) # 510 points for Reflux Ratio
B_values = np.linspace(0.8, 2.0, 510) # 510 points for Boilup Ratio, a typical operating range
xF_values = np.linspace(0.2, 0.95, 510) # 510 points for Feed Mole Fraction of Light Key (Ethanol)
F_values = np.linspace(base_F * 0.7, base_F * 1.3, 510) # ±30% around base_F

# For N, let's keep it simple for convergence, and use the base_N
# If you want to vary N, you would need to regenerate the column object or call SetNumberOfStages for each N.
# For simplicity and to ensure convergence, we will fix N to base_N for these 510 data points as requested by "prefer for convergence accordingly".
N_values = [base_N] * 510 # Fixed number of stages for all 510 simulations

# --- Store Results ---
results = []
simulation_count = 0

# --- Run Multiple Simulations ---
for i in range(510): # Collect 510 data points
    print(f"\n--- Running Simulation {i+1}/510 ---")

    # Set varying inputs for the current simulation
    current_R = R_values[i]
    current_B = B_values[i]
    current_xF_ethanol = xF_values[i] # Ethanol is light key
    current_xF_water = 1.0 - current_xF_ethanol
    current_F = F_values[i]
    current_N = N_values[i] # This will be base_N

    # Apply settings to the DWSIM objects
    column.SetNumberOfStages(int(current_N)) # Ensure N is an integer

    feed.SetOverallComposition(Array[float]([current_xF_water, current_xF_ethanol]))
    feed.SetMolarFlow(current_F)

    column.SetCondenserSpec("Reflux Ratio", float(current_R), "")
    column.SetReboilerSpec("Stream_Ratio", float(current_B), "") # Boilup ratio

    # Request a calculation
    errors = manager.CalculateFlowsheet4(myflowsheet)

    if errors is None or len(errors) == 0:
        simulation_count += 1
        print(f"Simulation {simulation_count} converged successfully!")

        # Get outputs
        cduty = column.CondenserDuty
        rduty = column.ReboilerDuty # Reboiler Duty (QR)

        dtemp = dist.GetTemperature()
        dflow = dist.GetMolarFlow()
        btemp = bottoms.GetTemperature()
        bflow = bottoms.GetMolarFlow()

        distcomp = dist.GetOverallComposition()
        xD_ethanol = distcomp[1] # Distillate mole fraction of Ethanol (light key)

        # Record inputs and outputs
        results.append({
            "R": current_R,
            "B": current_B,
            "xF_Ethanol": current_xF_ethanol,
            "F": current_F,
            "N": current_N,
            "xD_Ethanol": xD_ethanol,
            "QR_kW": rduty
        })

        # Print results for current simulation
        print(f"Inputs: R={current_R:.2f}, B={current_B:.2f}, xF_Ethanol={current_xF_ethanol:.2f}, F={current_F:.2f} mol/s, N={current_N}")
        print(f"Outputs: xD_Ethanol={xD_ethanol:.4f}, QR={rduty:.2f} kW")

    else:
        print(f"Simulation {i+1} FAILED to converge. Errors: {errors}")
        # You might want to store NaN or skip this data point if it fails to converge
        results.append({
            "R": current_R,
            "B": current_B,
            "xF_Ethanol": current_xF_ethanol,
            "F": current_F,
            "N": current_N,
            "xD_Ethanol": np.nan, # Not a Number for failed convergence
            "QR_kW": np.nan
        })

# --- Display and Save Results ---
print("\n--- All Simulations Complete ---")
df_results = pd.DataFrame(results)
print("\nCollected Data Points:")
print(df_results)

# Save results to a CSV file
results_file = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "distillation_sim_results.csv")
df_results.to_csv(results_file, index=False)
print(f"\nResults saved to: {results_file}")

# --- PFD Generation (Optional, for the last successful simulation) ---


if simulation_count > 0: # Only generate PFD if at least one simulation was successful
    print("\nGenerating PFD for the last successful simulation...")
    clr.AddReference(dwsimpath + "SkiaSharp.dll")
    clr.AddReference("System.Drawing")

    from SkiaSharp import SKBitmap, SKImage, SKCanvas, SKEncodedImageFormat
    from System.IO import MemoryStream
    from System.Drawing import Image
    from System.Drawing.Imaging import ImageFormat

    PFDSurface = myflowsheet.GetSurface()

    imgwidth = 1024
    imgheight = 768

    bmp = SKBitmap(imgwidth, imgheight)
    canvas = SKCanvas(bmp)
    PFDSurface.Center(imgwidth, imgheight)
    PFDSurface.ZoomAll(imgwidth, imgheight)
    PFDSurface.UpdateCanvas(canvas)
    d = SKImage.FromBitmap(bmp).Encode(SKEncodedImageFormat.Png, 100)
    str_stream = MemoryStream() # Renamed to avoid conflict with System.String
    d.SaveTo(str_stream)
    image = Image.FromStream(str_stream)
    imgPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "pfd_last_sim.png")
    image.Save(imgPath, ImageFormat.Png)
    str_stream.Dispose()
    canvas.Dispose()
    bmp.Dispose()

    from PIL import Image as PILImage # Renamed to avoid conflict

    pil_im = PILImage.open(imgPath)
    pil_im.show()
    print(f"PFD for the last successful simulation saved to: {imgPath}")



