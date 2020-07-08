var currentDiv = null
var beforeDiv = null

function showForm(selectElement)
{
    currentDiv = document.getElementById(selectElement.value)

    if(beforeDiv !== null)
    {
        beforeDiv.style.display="none"
        inputs = beforeDiv.querySelectorAll("input[type=number]")
        for(i=0; i<inputs.length; i++)
        {
            inputs[i].disabled = true
        }
        beforeDiv.querySelector("input[type=text]").disabled = true
        beforeDiv.querySelector("select").disabled = true
        beforeDiv.querySelector("input[type=checkbox]").checked = false
    }

    beforeDiv = currentDiv

    chooseModelDiv = document.getElementsByClassName("custom-control")[0]
    chooseModelCheckBox = chooseModelDiv.querySelector("input")
    chooseModelCheckBox.checked = false

    if(currentDiv == null)
    {
         chooseModelDiv.style.display = "none"
         return null
    }

    chooseModelDiv.style.display = "block"
    currentDiv.style.display = "block"

    showModels(chooseModelCheckBox)
}

function showModels(checkElement)
{
    if(checkElement.checked == true)
    {
        fieldSet = currentDiv.querySelector("fieldset")
        fieldSet.style.display = "none"
        inputs = fieldSet.querySelectorAll("input[type=number]")
        for (i=0; i<inputs.length; i++)
        {
            inputs[i].disabled = true
        }
        fieldSet.querySelector("input[type=text]").disabled = true

        selectDiv = currentDiv.querySelector(".input-group")
        selectDiv.style.display = "block"
        selectDiv.querySelector("select").disabled = false
    }
    else
    {
        fieldSet = currentDiv.querySelector("fieldset")
        fieldSet.style.display = "block"
        inputs = fieldSet.querySelectorAll("input[type=number]")
        for (i=0; i<inputs.length; i++)
        {
            inputs[i].disabled = false
        }
        saveModel(currentDiv.querySelector(".custom-switch input"))

        selectDiv = currentDiv.querySelector(".input-group")
        selectDiv.style.display = "none"
        selectDiv.querySelector("select").disabled = true
    }
}

function saveModel(checkElement)
{
    if(checkElement.checked == true)
    {
        modelNameDiv = currentDiv.querySelector("fieldset>div:last-child")
        modelNameDiv.style.display = "flex"
        modelNameDiv.querySelector("input").disabled = false
    }
    else
    {
        modelNameDiv = currentDiv.querySelector("fieldset>div:last-child")
        modelNameDiv.style.display = "none"
        modelNameDiv.querySelector("input").disabled = true
    }
}
