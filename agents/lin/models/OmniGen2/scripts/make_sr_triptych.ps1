param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$HrPath,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string]$OutputLabel
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$inputImage = [System.Drawing.Image]::FromFile((Resolve-Path $InputPath))
$outputImage = [System.Drawing.Image]::FromFile((Resolve-Path $OutputPath))
$hrImage = [System.Drawing.Image]::FromFile((Resolve-Path $HrPath))

try {
    $panelWidth = $outputImage.Width
    $panelHeight = $outputImage.Height
    $labelHeight = 56
    $canvas = New-Object System.Drawing.Bitmap ($panelWidth * 3), ($panelHeight + $labelHeight)
    try {
        $graphics = [System.Drawing.Graphics]::FromImage($canvas)
        try {
            $graphics.Clear([System.Drawing.Color]::White)
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
            $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality

            $graphics.DrawImage($inputImage, 0, $labelHeight, $panelWidth, $panelHeight)
            $graphics.DrawImage($outputImage, $panelWidth, $labelHeight, $panelWidth, $panelHeight)
            $graphics.DrawImage($hrImage, $panelWidth * 2, $labelHeight, $panelWidth, $panelHeight)

            $font = New-Object System.Drawing.Font "Arial", 18, ([System.Drawing.FontStyle]::Bold)
            $brush = [System.Drawing.Brushes]::Black
            $format = New-Object System.Drawing.StringFormat
            $format.Alignment = [System.Drawing.StringAlignment]::Center
            $format.LineAlignment = [System.Drawing.StringAlignment]::Center
            try {
                $labels = @("Input (bicubic display)", $OutputLabel, "HR reference")
                for ($i = 0; $i -lt 3; $i++) {
                    $rect = New-Object System.Drawing.RectangleF ($i * $panelWidth), 0, $panelWidth, $labelHeight
                    $graphics.DrawString($labels[$i], $font, $brush, $rect, $format)
                }
            }
            finally {
                $format.Dispose()
                $font.Dispose()
            }
        }
        finally {
            $graphics.Dispose()
        }

        $destinationDirectory = Split-Path -Parent $Destination
        if ($destinationDirectory) {
            New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        }
        $canvas.Save($Destination, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $canvas.Dispose()
    }
}
finally {
    $inputImage.Dispose()
    $outputImage.Dispose()
    $hrImage.Dispose()
}

Write-Output $Destination
