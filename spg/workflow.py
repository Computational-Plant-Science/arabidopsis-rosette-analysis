# WORKFLOW_CONFIG: Configuration information for the Plant IT server
WORKFLOW_CONFIG = {
    ###
    # These values should not need changed.
    ###

    #Human readable workflow name
    "name": "Plant Image Analysis",

    #Human readable description
    "description": "",

    #Location of icon to show for the workflow.
    # This path is relative to the django/static/ directory.
    # Due to the way django deals with static files, please
    # do not place anything outside of 'workflows/plant_image_analysis'
    "icon_loc": "workflows/plant_image_analysis/icon.png",

    #The cluster side api version that this workflow requires.
    # Provided for future compatibility. Currently, the only valid version
    # is 0.1.
    "api_version": 0.1,

    #The computer readable app name. Must be a valid python module name
    "app_name": "plant_image_analysis",

    ###
    # These values you may need to change
    ###

    #The url to the singularity container in which to run
    # process_sample(). The provided singularity container must have
    # python3 installed. And it must be executable using the
    # 'python3' command.
    "singularity_url": "singularity shell shub://lsx1980/plant-image-analysis",
}

#Defines the arguments that are passed to process_sample.
#See docs for more details
parameters = [
    {
        #unique group name, must be a valid python variable name
        'id': 'settings',

        #Human readable name, shown in website UI
        'name': 'Workflow Settings',

        #A list of parameters within this group. These are converted to
        # fields in the website UI. Each parameter is represented by a
        # python dictionary.
        'params':[
            {
                #unique parameter name, must be a valid python variable name
                'id': 'filetype',

                #Type of value this field will handle
                #Valid types are: bool, float, and int
                'type': 'str',

                #Initial value the field will have.
                #Must be compatible with the type of field
                'initial': "jpg",

                #Human readable name, shown in website UI
                'name': 'image_filetype',

                #A human readable description of what this parameter does
                #shown to the users in the website UI
                'description': 'Input image file type'
            },
        
        ]
    }
]
