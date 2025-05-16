; Flag to track blocking
blockClick := false

; Toggle left click blocking with F11
F11::
    blockClick := !blockClick
    if (blockClick) {
        ToolTip, 🛑 Left click is DISABLED
    } else {
        ToolTip, ✅ Left click is ENABLED
    }
    SetTimer, RemoveToolTip, -1000
return

; Intercept and block left-click down
LButton::
    if (blockClick)
        return  ; Block the click
    Click  ; Allow click
return

; Optional: block left-click up as well
LButton Up::
    if (blockClick)
        return
return

RemoveToolTip:
    ToolTip
return
